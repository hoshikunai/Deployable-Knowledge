import {
	TRAINING_EARLY_STOPPING_PATIENCE,
	TRAINING_L2_REGULARIZATION,
	TRAINING_LEARNING_RATE,
	TRAINING_LOSS_TOLERANCE,
	TRAINING_MAX_EPOCHS
} from './retrieval-training-constants';
import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';

export interface LinearRetrievalRankerParameters {
	weights: number[];
	intercept: number;
}

export interface TrainedLinearRetrievalRanker extends LinearRetrievalRankerParameters {
	epochs: number;
	trainingLoss: number;
}

function buildRatingWeights(
	examples: PreparedRetrievalTrainingExample[]
): Map<PreparedRetrievalTrainingExample['rating'], number> {
	const ratingCounts = new Map<PreparedRetrievalTrainingExample['rating'], number>();

	for (const example of examples) {
		ratingCounts.set(example.rating, (ratingCounts.get(example.rating) ?? 0) + 1);
	}

	const representedRatingCount = ratingCounts.size;
	return new Map(
		[...ratingCounts].map(([rating, count]) => [
			rating,
			examples.length / (representedRatingCount * count)
		])
	);
}

export function predictRetrievalRating(
	ranker: LinearRetrievalRankerParameters,
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

export function trainLinearRetrievalRanker(
	examples: PreparedRetrievalTrainingExample[]
): TrainedLinearRetrievalRanker {
	if (examples.length === 0) {
		throw new Error('Cannot train a retrieval ranker without examples.');
	}

	const featureCount = examples[0].features.length;
	const weights = new Array<number>(featureCount).fill(0);
	let intercept = 0;
	const ratingWeights = buildRatingWeights(examples);
	const exampleWeights = examples.map((example) => {
		const weight = ratingWeights.get(example.rating);
		if (weight === undefined) {
			throw new Error(`Training rating ${example.rating} does not have a class weight.`);
		}
		return weight;
	});
	const totalExampleWeight = exampleWeights.reduce((total, weight) => total + weight, 0);

	let bestWeights = [...weights];
	let bestIntercept = intercept;
	let bestLoss = Number.POSITIVE_INFINITY;
	let epochsWithoutImprovement = 0;
	let completedEpochs = 0;

	for (let epoch = 1; epoch <= TRAINING_MAX_EPOCHS; epoch += 1) {
		const weightGradients = new Array<number>(featureCount).fill(0);
		let interceptGradient = 0;
		let squaredErrorTotal = 0;

		for (const [exampleIndex, example] of examples.entries()) {
			const prediction = predictRetrievalRating({ weights, intercept }, example.features);
			const error = prediction - example.target;
			const exampleWeight = exampleWeights[exampleIndex];

			squaredErrorTotal += exampleWeight * error * error;
			interceptGradient += 2 * exampleWeight * error;

			for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
				weightGradients[featureIndex] += 2 * exampleWeight * error * example.features[featureIndex];
			}
		}

		const meanSquaredError = squaredErrorTotal / totalExampleWeight;
		const regularizationLoss =
			TRAINING_L2_REGULARIZATION * weights.reduce((total, weight) => total + weight * weight, 0);
		const loss = meanSquaredError + regularizationLoss;

		if (!Number.isFinite(loss)) {
			throw new Error('Retrieval training produced a non-finite loss.');
		}

		completedEpochs = epoch;

		if (bestLoss - loss > TRAINING_LOSS_TOLERANCE) {
			bestLoss = loss;
			bestWeights = [...weights];
			bestIntercept = intercept;
			epochsWithoutImprovement = 0;
		} else {
			epochsWithoutImprovement += 1;
		}

		if (epochsWithoutImprovement >= TRAINING_EARLY_STOPPING_PATIENCE) {
			break;
		}

		intercept -= TRAINING_LEARNING_RATE * (interceptGradient / totalExampleWeight);

		for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
			const dataGradient = weightGradients[featureIndex] / totalExampleWeight;
			const regularizationGradient = 2 * TRAINING_L2_REGULARIZATION * weights[featureIndex];

			weights[featureIndex] -= TRAINING_LEARNING_RATE * (dataGradient + regularizationGradient);
		}

		if (!Number.isFinite(intercept) || !weights.every(Number.isFinite)) {
			throw new Error('Retrieval training produced non-finite model parameters.');
		}
	}

	return {
		weights: bestWeights,
		intercept: bestIntercept,
		epochs: completedEpochs,
		trainingLoss: bestLoss
	};
}
