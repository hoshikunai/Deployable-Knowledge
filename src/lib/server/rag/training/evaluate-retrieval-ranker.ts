import { blendLearnedRanking } from '$lib/server/rag/search/blend-learned-ranking';
import { buildRetrievalPreferencePairs } from './build-retrieval-preference-pairs';
import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import type { RetrievalTrainingEvaluation } from './retrieval-model.types';
import {
	LEARNED_RANKING_BLEND_WEIGHT,
	RETRIEVAL_RANKING_STRATEGY
} from './retrieval-training-constants';

export type { RetrievalTrainingEvaluation } from './retrieval-model.types';

const NDCG_LIMIT = 5;
const SCORE_TIE_TOLERANCE = 1e-9;

export interface ScoredRetrievalTrainingExample {
	example: PreparedRetrievalTrainingExample;
	learnedScore: number;
}

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

function buildRankingGroupKey(example: PreparedRetrievalTrainingExample): string {
	return `${example.impressionId}\u0000${example.retrievalMode}`;
}

export function evaluateRetrievalRanker(
	scoredExamples: ScoredRetrievalTrainingExample[],
	crossValidationFolds: number
): RetrievalTrainingEvaluation {
	if (scoredExamples.length === 0) {
		throw new Error('Cannot evaluate a retrieval ranker without scored examples.');
	}

	if (
		!Number.isInteger(crossValidationFolds) ||
		crossValidationFolds < 2 ||
		!scoredExamples.every(({ learnedScore }) => Number.isFinite(learnedScore))
	) {
		throw new Error('Retrieval ranker evaluation received invalid cross-validation results.');
	}

	const learnedScores = new Map(
		scoredExamples.map(({ example, learnedScore }) => [example.feedbackId, learnedScore])
	);
	const preferencePairs = buildRetrievalPreferencePairs(
		scoredExamples.map(({ example }) => example)
	);
	const totalPairWeight = preferencePairs.reduce((total, pair) => total + pair.weight, 0);
	const correctPairWeight = preferencePairs.reduce((total, pair) => {
		const preferredScore = learnedScores.get(pair.preferredFeedbackId);
		const rejectedScore = learnedScores.get(pair.rejectedFeedbackId);

		if (preferredScore === undefined || rejectedScore === undefined) {
			throw new Error('A cross-validation preference pair is missing a learned score.');
		}

		const difference = preferredScore - rejectedScore;
		if (difference > SCORE_TIE_TOLERANCE) return total + pair.weight;
		if (Math.abs(difference) <= SCORE_TIE_TOLERANCE) return total + pair.weight * 0.5;
		return total;
	}, 0);

	if (preferencePairs.length === 0 || totalPairWeight <= 0) {
		throw new Error('Cannot evaluate a pairwise retrieval ranker without preference pairs.');
	}

	const groups = new Map<string, ScoredRetrievalTrainingExample[]>();

	for (const scoredExample of scoredExamples) {
		const groupKey = buildRankingGroupKey(scoredExample.example);
		const group = groups.get(groupKey) ?? [];
		group.push(scoredExample);
		groups.set(groupKey, group);
	}

	const rankingGroupScores: Array<{ baseline: number; trained: number }> = [];

	for (const group of groups.values()) {
		const distinctRatings = new Set(group.map(({ example }) => example.rating));
		if (group.length < 2 || distinctRatings.size < 2) continue;

		const ideal = group
			.map(({ example }) => example)
			.sort((left, right) => right.rating - left.rating || left.baseRank - right.baseRank);
		const baseline = [...group].sort(
			(left, right) => left.example.baseRank - right.example.baseRank
		);
		const trained = blendLearnedRanking(
			baseline.map(({ example, learnedScore }) => ({
				value: example,
				learnedScore
			})),
			LEARNED_RANKING_BLEND_WEIGHT
		).map(({ value }) => value);

		rankingGroupScores.push({
			baseline: calculateNdcg(
				baseline.map(({ example }) => example),
				ideal
			),
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
	const ndcgImprovement =
		baselineNdcgAt5 === null || trainedNdcgAt5 === null ? null : trainedNdcgAt5 - baselineNdcgAt5;
	const rankingGroupDeltas = rankingGroupScores.map(({ baseline, trained }) => trained - baseline);
	const improvedRankingGroups = rankingGroupDeltas.filter(
		(delta) => delta > SCORE_TIE_TOLERANCE
	).length;
	const degradedRankingGroups = rankingGroupDeltas.filter(
		(delta) => delta < -SCORE_TIE_TOLERANCE
	).length;
	const tiedRankingGroups =
		rankingGroupDeltas.length - improvedRankingGroups - degradedRankingGroups;

	return {
		rankingStrategy: RETRIEVAL_RANKING_STRATEGY,
		blendWeight: LEARNED_RANKING_BLEND_WEIGHT,
		crossValidationFolds,
		pairwiseAccuracy: correctPairWeight / totalPairWeight,
		evaluatedPairs: preferencePairs.length,
		baselineNdcgAt5,
		trainedNdcgAt5,
		ndcgImprovement,
		evaluatedExamples: scoredExamples.length,
		evaluatedRankingGroups: rankingGroupScores.length,
		improvedRankingGroups,
		degradedRankingGroups,
		tiedRankingGroups,
		worstRankingGroupNdcgDelta:
			rankingGroupDeltas.length > 0 ? Math.min(...rankingGroupDeltas) : null
	};
}
