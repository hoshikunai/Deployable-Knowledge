import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import {
	predictRetrievalRating,
	type LinearRetrievalRankerParameters
} from './train-linear-retrieval-ranker';

const NDCG_LIMIT = 5;

export interface RetrievalTrainingEvaluation {
	meanAbsoluteError: number;
	baselineNdcgAt5: number | null;
	trainedNdcgAt5: number | null;
	ndcgImprovement: number | null;
	evaluatedExamples: number;
	evaluatedRankingGroups: number;
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

	const baselineScores: number[] = [];
	const trainedScores: number[] = [];

	for (const group of groups.values()) {
		if (group.length < 2) continue;

		const ideal = [...group].sort((left, right) => right.rating - left.rating);
		const baseline = [...group].sort((left, right) => left.baseRank - right.baseRank);
		const trained = [...group].sort((left, right) => {
			const rightPrediction = predictRetrievalRating(ranker, right.features);
			const leftPrediction = predictRetrievalRating(ranker, left.features);
			const predictionDifference = rightPrediction - leftPrediction;

			if (predictionDifference !== 0) return predictionDifference;
			return left.baseRank - right.baseRank;
		});

		baselineScores.push(calculateNdcg(baseline, ideal));
		trainedScores.push(calculateNdcg(trained, ideal));
	}

	const baselineNdcgAt5 =
		baselineScores.length > 0
			? baselineScores.reduce((total, score) => total + score, 0) / baselineScores.length
			: null;

	const trainedNdcgAt5 =
		trainedScores.length > 0
			? trainedScores.reduce((total, score) => total + score, 0) / trainedScores.length
			: null;

	let ndcgImprovement: number | null = null;
	if (baselineNdcgAt5 !== null && trainedNdcgAt5 !== null) {
		ndcgImprovement = trainedNdcgAt5 - baselineNdcgAt5;
	}

	return {
		meanAbsoluteError: absoluteErrorTotal / validationExamples.length,
		baselineNdcgAt5,
		trainedNdcgAt5,
		ndcgImprovement,
		evaluatedExamples: validationExamples.length,
		evaluatedRankingGroups: baselineScores.length
	};
}
