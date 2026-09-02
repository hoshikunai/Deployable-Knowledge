import type { RetrievalPreferencePair } from './build-retrieval-preference-pairs';
import {
	TRAINING_EARLY_STOPPING_PATIENCE,
	TRAINING_L2_REGULARIZATION,
	TRAINING_LEARNING_RATE,
	TRAINING_LOSS_TOLERANCE,
	TRAINING_MAX_EPOCHS
} from './retrieval-training-constants';

export interface PairwiseRetrievalRankerParameters {
	weights: number[];
	intercept: number;
}

export interface TrainedPairwiseRetrievalRanker extends PairwiseRetrievalRankerParameters {
	epochs: number;
	trainingLoss: number;
}

export function predictRetrievalUtility(
	ranker: PairwiseRetrievalRankerParameters,
	features: number[]
): number {
	if (ranker.weights.length !== features.length) {
		throw new Error('The retrieval ranker and feature vector have different dimensions.');
	}

	return features.reduce(
		(prediction, feature, index) => prediction + feature * ranker.weights[index],
		ranker.intercept
	);
}

function pairMargin(weights: number[], pair: RetrievalPreferencePair): number {
	return pair.featureDifferences.reduce(
		(margin, difference, index) => margin + difference * weights[index],
		0
	);
}

function logisticLoss(margin: number): number {
	if (margin >= 0) return Math.log1p(Math.exp(-margin));
	return -margin + Math.log1p(Math.exp(margin));
}

function logisticLossDerivative(margin: number): number {
	if (margin >= 0) {
		const inverseExponent = Math.exp(-margin);
		return -inverseExponent / (1 + inverseExponent);
	}

	return -1 / (1 + Math.exp(margin));
}

export function trainPairwiseRetrievalRanker(
	pairs: RetrievalPreferencePair[]
): TrainedPairwiseRetrievalRanker {
	if (pairs.length === 0) {
		throw new Error('Cannot train a pairwise retrieval ranker without preference pairs.');
	}

	const featureCount = pairs[0].featureDifferences.length;
	if (
		featureCount === 0 ||
		!pairs.every((pair) => pair.featureDifferences.length === featureCount)
	) {
		throw new Error('Retrieval preference pairs have inconsistent feature dimensions.');
	}

	const totalPairWeight = pairs.reduce((total, pair) => total + pair.weight, 0);
	if (!Number.isFinite(totalPairWeight) || totalPairWeight <= 0) {
		throw new Error('Retrieval preference pairs do not have a valid total weight.');
	}

	const weights = new Array<number>(featureCount).fill(0);
	let bestWeights = [...weights];
	let bestLoss = Number.POSITIVE_INFINITY;
	let epochsWithoutImprovement = 0;
	let completedEpochs = 0;

	for (let epoch = 1; epoch <= TRAINING_MAX_EPOCHS; epoch += 1) {
		const gradients = new Array<number>(featureCount).fill(0);
		let weightedLoss = 0;

		for (const pair of pairs) {
			const margin = pairMargin(weights, pair);
			const lossDerivative = logisticLossDerivative(margin);
			weightedLoss += pair.weight * logisticLoss(margin);

			for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
				gradients[featureIndex] +=
					pair.weight * lossDerivative * pair.featureDifferences[featureIndex];
			}
		}

		const regularizationLoss =
			TRAINING_L2_REGULARIZATION * weights.reduce((total, weight) => total + weight * weight, 0);
		const loss = weightedLoss / totalPairWeight + regularizationLoss;

		if (!Number.isFinite(loss)) {
			throw new Error('Pairwise retrieval training produced a non-finite loss.');
		}

		completedEpochs = epoch;

		if (bestLoss - loss > TRAINING_LOSS_TOLERANCE) {
			bestLoss = loss;
			bestWeights = [...weights];
			epochsWithoutImprovement = 0;
		} else {
			epochsWithoutImprovement += 1;
		}

		if (epochsWithoutImprovement >= TRAINING_EARLY_STOPPING_PATIENCE) break;

		for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
			const dataGradient = gradients[featureIndex] / totalPairWeight;
			const regularizationGradient = 2 * TRAINING_L2_REGULARIZATION * weights[featureIndex];
			weights[featureIndex] -= TRAINING_LEARNING_RATE * (dataGradient + regularizationGradient);
		}

		if (!weights.every(Number.isFinite)) {
			throw new Error('Pairwise retrieval training produced non-finite model parameters.');
		}
	}

	return {
		weights: bestWeights,
		intercept: 0,
		epochs: completedEpochs,
		trainingLoss: bestLoss
	};
}
