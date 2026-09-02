import { buildQueryGroupedCrossValidationFolds } from './build-query-grouped-cross-validation-folds';
import { buildRetrievalPreferencePairs } from './build-retrieval-preference-pairs';
import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import {
	evaluateRetrievalRanker,
	type RetrievalTrainingEvaluation,
	type ScoredRetrievalTrainingExample
} from './evaluate-retrieval-ranker';
import {
	fitRetrievalFeatureScaler,
	scaleRetrievalTrainingExamples
} from './retrieval-feature-scaler';
import { CROSS_VALIDATION_FOLD_COUNT } from './retrieval-training-constants';
import { RetrievalTrainingReadinessError } from './select-retrieval-training-cohort';
import {
	predictRetrievalUtility,
	trainPairwiseRetrievalRanker
} from './train-pairwise-retrieval-ranker';

export function crossValidateRetrievalRanker(
	examples: PreparedRetrievalTrainingExample[]
): RetrievalTrainingEvaluation {
	const folds = buildQueryGroupedCrossValidationFolds(examples);
	const scoredExamples: ScoredRetrievalTrainingExample[] = [];

	for (const fold of folds) {
		const scaler = fitRetrievalFeatureScaler(fold.training);
		const scaledTraining = scaleRetrievalTrainingExamples(fold.training, scaler);
		const scaledValidation = scaleRetrievalTrainingExamples(fold.validation, scaler);
		const trainingPairs = buildRetrievalPreferencePairs(scaledTraining);

		if (trainingPairs.length === 0) {
			throw new RetrievalTrainingReadinessError(
				`Retrieval cross-validation fold ${fold.index + 1} has no training preferences.`
			);
		}

		const ranker = trainPairwiseRetrievalRanker(trainingPairs);
		scoredExamples.push(
			...scaledValidation.map((example) => ({
				example,
				learnedScore: predictRetrievalUtility(ranker, example.features)
			}))
		);
	}

	return evaluateRetrievalRanker(scoredExamples, CROSS_VALIDATION_FOLD_COUNT);
}
