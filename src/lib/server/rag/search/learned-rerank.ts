import { RetrievalMode } from '$lib/enums';
import { buildRetrievalFeatureVector } from '$lib/server/rag/training/build-retrieval-training-features';
import { scaleRetrievalFeatureVector } from '$lib/server/rag/training/retrieval-feature-scaler';
import { LEARNED_RANKING_BLEND_WEIGHT } from '$lib/server/rag/training/retrieval-training-constants';
import { predictRetrievalRating } from '$lib/server/rag/training/train-linear-retrieval-ranker';
import { loadActiveRetrievalModel } from './active-retrieval-model';
import { blendLearnedRanking } from './blend-learned-ranking';
import type { RetrievalScoreMaps, ScoredSearchMatch } from './search-shared';

export interface LearnedRerankResult {
	matches: ScoredSearchMatch[];
	modelId: string | null;
	scores: Map<string, number>;
}

export async function rerankWithActiveRetrievalModel(
	retrievalMode: RetrievalMode,
	matches: ScoredSearchMatch[],
	scoreMaps: RetrievalScoreMaps
): Promise<LearnedRerankResult> {
	if (retrievalMode !== RetrievalMode.HYBRID || matches.length === 0) {
		return {
			matches,
			modelId: null,
			scores: new Map()
		};
	}

	const model = await loadActiveRetrievalModel();

	if (!model) {
		return {
			matches,
			modelId: null,
			scores: new Map()
		};
	}

	try {
		const candidates = matches.map((match, originalIndex) => {
			const features = buildRetrievalFeatureVector({
				retrievalMode,
				baseRank: originalIndex + 1,
				semanticScore: scoreMaps.semantic.get(match.chunkId) ?? null,
				bm25Score: scoreMaps.bm25.get(match.chunkId) ?? null,
				crossEncoderScore: scoreMaps.crossEncoder.get(match.chunkId) ?? null,
				baseScore: match.score
			});

			const scaledFeatures = scaleRetrievalFeatureVector(features, {
				means: model.means,
				standardDeviations: model.standardDeviations
			});

			const learnedScore = predictRetrievalRating(
				{
					weights: model.weights,
					intercept: model.intercept
				},
				scaledFeatures
			);

			if (!Number.isFinite(learnedScore)) {
				throw new Error(`Learned score for chunk ${match.chunkId} is not finite.`);
			}

			return {
				value: match,
				learnedScore
			};
		});
		const ranked = blendLearnedRanking(candidates, LEARNED_RANKING_BLEND_WEIGHT);

		return {
			matches: ranked.map(({ value, blendedScore }) => ({
				...value,
				score: blendedScore
			})),
			modelId: model.id,
			scores: new Map(ranked.map(({ value, learnedScore }) => [value.chunkId, learnedScore]))
		};
	} catch (error) {
		console.warn('[Retrieval] Learned reranking failed; using the base ranking.', error);
		return {
			matches,
			modelId: null,
			scores: new Map()
		};
	}
}
