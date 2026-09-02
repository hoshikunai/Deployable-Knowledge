import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import type { RetrievalTrainingEvaluation } from './retrieval-model.types';
import { blendLearnedRanking } from '$lib/server/rag/search/blend-learned-ranking';
import {
	predictRetrievalRating,
	type LinearRetrievalRankerParameters
} from './train-linear-retrieval-ranker';
import {
	LEARNED_RANKING_BLEND_WEIGHT,
	RETRIEVAL_RANKING_STRATEGY
} from './retrieval-training-constants';

export type { RetrievalTrainingEvaluation } from './retrieval-model.types';

const NDCG_LIMIT = 5;
const NDCG_TIE_TOLERANCE = 1e-9;

function discountedCumulativeGain(examples: PreparedRetrievalTrainingExample[]): number {
	return examples.slice(0, NDCG_LIMIT).reduce((total, example, index) => {
		const gain = 2 ** example.rating - 1;
		const discount = Math.log2(index + 2);
		return total + gain / discount;
	}, 0);
}

function calculateNdcg(
	ordered: PreparedRetrievalTrainingExample[],
	ideal: PreparedRetrievalTrainingExample[]
): number {
	const idealScore = discountedCumulativeGain(ideal);
	if (idealScore === 0) return 0;

	return discountedCumulativeGain(ordered) / idealScore;
}

export function evaluateRetrievalRanker(
	ranker: LinearRetrievalRankerParameters,
	validationExamples: PreparedRetrievalTrainingExample[]
): RetrievalTrainingEvaluation {
	if (validationExamples.length === 0) {
		throw new Error('Cannot evaluate a retrieval ranker without validation examples.');
	}

	const absoluteErrorTotal = validationExamples.reduce((total, example) => {
		const prediction = predictRetrievalRating(ranker, example.features);
		return total + Math.abs(prediction - example.target);
	}, 0);

	const groups = new Map<string, PreparedRetrievalTrainingExample[]>();

	for (const example of validationExamples) {
		const groupKey = `${example.queryHash}\u0000${example.retrievalMode}`;
		const group = groups.get(groupKey) ?? [];
		group.push(example);
		groups.set(groupKey, group);
	}

	const rankingGroupScores: Array<{ baseline: number; trained: number }> = [];

	for (const group of groups.values()) {
		if (group.length < 2) continue;

		const ideal = [...group].sort((left, right) => right.rating - left.rating);
		const baseline = [...group].sort((left, right) => left.baseRank - right.baseRank);
		const trained = blendLearnedRanking(
			baseline.map((example) => ({
				value: example,
				learnedScore: predictRetrievalRating(ranker, example.features)
			})),
			LEARNED_RANKING_BLEND_WEIGHT
		).map(({ value }) => value);

		rankingGroupScores.push({
			baseline: calculateNdcg(baseline, ideal),
			trained: calculateNdcg(trained, ideal)
		});
	}

	const baselineNdcgAt5 =
		rankingGroupScores.length > 0
			? rankingGroupScores.reduce((total, scores) => total + scores.baseline, 0) /
				rankingGroupScores.length
			: null;

	const trainedNdcgAt5 =
		rankingGroupScores.length > 0
			? rankingGroupScores.reduce((total, scores) => total + scores.trained, 0) /
				rankingGroupScores.length
			: null;

	let ndcgImprovement: number | null = null;
	if (baselineNdcgAt5 !== null && trainedNdcgAt5 !== null) {
		ndcgImprovement = trainedNdcgAt5 - baselineNdcgAt5;
	}

	const rankingGroupDeltas = rankingGroupScores.map(({ baseline, trained }) => trained - baseline);
	const improvedRankingGroups = rankingGroupDeltas.filter(
		(delta) => delta > NDCG_TIE_TOLERANCE
	).length;
	const degradedRankingGroups = rankingGroupDeltas.filter(
		(delta) => delta < -NDCG_TIE_TOLERANCE
	).length;
	const tiedRankingGroups =
		rankingGroupDeltas.length - improvedRankingGroups - degradedRankingGroups;

	return {
		rankingStrategy: RETRIEVAL_RANKING_STRATEGY,
		blendWeight: LEARNED_RANKING_BLEND_WEIGHT,
		meanAbsoluteError: absoluteErrorTotal / validationExamples.length,
		baselineNdcgAt5,
		trainedNdcgAt5,
		ndcgImprovement,
		evaluatedExamples: validationExamples.length,
		evaluatedRankingGroups: rankingGroupScores.length,
		improvedRankingGroups,
		degradedRankingGroups,
		tiedRankingGroups,
		worstRankingGroupNdcgDelta:
			rankingGroupDeltas.length > 0 ? Math.min(...rankingGroupDeltas) : null
	};
}
