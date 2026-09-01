import {
	RETRIEVAL_FEATURE_NAMES,
	type PreparedRetrievalTrainingExample
} from './build-retrieval-training-features';

const CONTINUOUS_FEATURE_INDEXES = new Set([0, 1, 3, 5, 7]);
const MINIMUM_STANDARD_DEVIATION = 1e-8;

export interface RetrievalFeatureScaler {
	means: number[];
	standardDeviations: number[];
}

function validateFeatureLength(features: number[]): void {
	if (features.length !== RETRIEVAL_FEATURE_NAMES.length) {
		throw new Error(
			`Expected ${RETRIEVAL_FEATURE_NAMES.length} retrieval features, received ${features.length}.`
		);
	}
}

export function fitRetrievalFeatureScaler(
	examples: PreparedRetrievalTrainingExample[]
): RetrievalFeatureScaler {
	if (examples.length === 0) {
		throw new Error('Cannot fit a feature scaler without training examples.');
	}

	for (const example of examples) {
		validateFeatureLength(example.features);
	}

	const means = RETRIEVAL_FEATURE_NAMES.map((_, featureIndex) => {
		if (!CONTINUOUS_FEATURE_INDEXES.has(featureIndex)) return 0;

		const total = examples.reduce((sum, example) => sum + example.features[featureIndex], 0);
		return total / examples.length;
	});

	const standardDeviations = RETRIEVAL_FEATURE_NAMES.map((_, featureIndex) => {
		if (!CONTINUOUS_FEATURE_INDEXES.has(featureIndex)) return 1;

		const squaredDifferenceTotal = examples.reduce((sum, example) => {
			const difference = example.features[featureIndex] - means[featureIndex];
			return sum + difference * difference;
		}, 0);

		const standardDeviation = Math.sqrt(squaredDifferenceTotal / examples.length);
		if (standardDeviation < MINIMUM_STANDARD_DEVIATION) return 1;

		return standardDeviation;
	});

	return {
		means,
		standardDeviations
	};
}

export function scaleRetrievalTrainingExamples(
	examples: PreparedRetrievalTrainingExample[],
	scaler: RetrievalFeatureScaler
): PreparedRetrievalTrainingExample[] {
	if (
		scaler.means.length !== RETRIEVAL_FEATURE_NAMES.length ||
		scaler.standardDeviations.length !== RETRIEVAL_FEATURE_NAMES.length
	) {
		throw new Error('The retrieval feature scaler has an invalid feature count.');
	}

	return examples.map((example) => {
		validateFeatureLength(example.features);

		const features = example.features.map(
			(value, featureIndex) =>
				(value - scaler.means[featureIndex]) / scaler.standardDeviations[featureIndex]
		);

		if (!features.every(Number.isFinite)) {
			throw new Error(`Scaling example ${example.feedbackId} produced a non-finite feature.`);
		}

		return {
			...example,
			features
		};
	});
}
