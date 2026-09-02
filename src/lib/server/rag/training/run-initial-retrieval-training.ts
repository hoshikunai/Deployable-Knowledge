import { HUMAN_EXPERT_FEEDBACK_SOURCE } from '$lib/constants';
import { RetrievalModelsRepository } from '$lib/server/repositories';
import type { RetrievalFeedbackSource } from '$lib/types';
import { buildRetrievalTrainingDataset } from './build-retrieval-training-dataset';
import { buildRetrievalPreferencePairs } from './build-retrieval-preference-pairs';
import {
	buildRetrievalTrainingFeatures,
	RETRIEVAL_FEATURE_NAMES
} from './build-retrieval-training-features';
import { crossValidateRetrievalRanker } from './cross-validate-retrieval-ranker';
import type { RetrievalTrainingEvaluation } from './evaluate-retrieval-ranker';
import {
	fitRetrievalFeatureScaler,
	scaleRetrievalTrainingExamples,
	type RetrievalFeatureScaler
} from './retrieval-feature-scaler';
import type { RetrievalTrainingHyperparameters } from './retrieval-model.types';
import {
	CROSS_VALIDATION_FOLD_COUNT,
	RETRIEVAL_FEATURE_VERSION,
	RETRIEVAL_PAIR_WEIGHTING_STRATEGY,
	RETRIEVAL_TRAINING_ALGORITHM,
	TRAINING_EARLY_STOPPING_PATIENCE,
	TRAINING_L2_REGULARIZATION,
	TRAINING_LEARNING_RATE,
	TRAINING_LOSS_TOLERANCE,
	TRAINING_MAX_EPOCHS
} from './retrieval-training-constants';
import type { RetrievalTrainingDatasetStats } from './retrieval-training.types';
import {
	RetrievalTrainingReadinessError,
	selectRetrievalTrainingCohort,
	type RetrievalTrainingCohort
} from './select-retrieval-training-cohort';
import {
	trainPairwiseRetrievalRanker,
	type TrainedPairwiseRetrievalRanker
} from './train-pairwise-retrieval-ranker';

export interface InitialRetrievalTrainingResult {
	runId: string;
	modelId: string;
	feedbackSource: RetrievalFeedbackSource;
	featureVersion: typeof RETRIEVAL_FEATURE_VERSION;
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	trainingExamples: number;
	validationExamples: number;
	trainingPairs: number;
	validationPairs: number;
	distinctQueries: number;
	datasetStats: RetrievalTrainingDatasetStats;
	scaler: RetrievalFeatureScaler;
	ranker: TrainedPairwiseRetrievalRanker;
	evaluation: RetrievalTrainingEvaluation;
	regularization: number;
}

export async function runInitialRetrievalTraining(
	feedbackSource: RetrievalFeedbackSource = HUMAN_EXPERT_FEEDBACK_SOURCE
): Promise<InitialRetrievalTrainingResult> {
	const dataset = await buildRetrievalTrainingDataset(feedbackSource);
	const cohort: RetrievalTrainingCohort = selectRetrievalTrainingCohort(dataset.examples);
	const preparedExamples = buildRetrievalTrainingFeatures(cohort.examples);
	const finalScaler = fitRetrievalFeatureScaler(preparedExamples);
	const scaledExamples = scaleRetrievalTrainingExamples(preparedExamples, finalScaler);
	const finalTrainingPairs = buildRetrievalPreferencePairs(scaledExamples);

	if (finalTrainingPairs.length === 0) {
		throw new RetrievalTrainingReadinessError(
			'Retrieval training needs at least one within-impression rating preference.'
		);
	}

	const hyperparameters: RetrievalTrainingHyperparameters = {
		algorithm: RETRIEVAL_TRAINING_ALGORITHM,
		learningRate: TRAINING_LEARNING_RATE,
		l2Regularization: TRAINING_L2_REGULARIZATION,
		maxEpochs: TRAINING_MAX_EPOCHS,
		earlyStoppingPatience: TRAINING_EARLY_STOPPING_PATIENCE,
		lossTolerance: TRAINING_LOSS_TOLERANCE,
		crossValidationFolds: CROSS_VALIDATION_FOLD_COUNT,
		pairWeightingStrategy: RETRIEVAL_PAIR_WEIGHTING_STRATEGY
	};

	const run = await RetrievalModelsRepository.startTrainingRun({
		feedbackSource,
		datasetVersion: dataset.version,
		featureVersion: RETRIEVAL_FEATURE_VERSION,
		embeddingModel: cohort.embeddingModel,
		rerankerModel: cohort.rerankerModel,
		scoringVersion: cohort.scoringVersion,
		trainingExamples: preparedExamples.length,
		validationExamples: preparedExamples.length,
		distinctQueries: cohort.distinctQueries,
		totalFeedback: dataset.stats.totalFeedback,
		attributedFeedback: dataset.stats.attributedFeedback,
		unattributedFeedback: dataset.stats.unattributedFeedback,
		inconsistentFeedback: dataset.stats.inconsistentFeedback,
		hyperparameters
	});

	try {
		const evaluation = crossValidateRetrievalRanker(preparedExamples);
		const ranker = trainPairwiseRetrievalRanker(finalTrainingPairs);

		const saved = await RetrievalModelsRepository.completeTrainingRun(run.id, {
			featureVersion: RETRIEVAL_FEATURE_VERSION,
			featureNames: RETRIEVAL_FEATURE_NAMES,
			scaler: finalScaler,
			ranker,
			evaluation,
			regularization: TRAINING_L2_REGULARIZATION
		});

		return {
			runId: saved.run.id,
			modelId: saved.model.id,
			feedbackSource,
			featureVersion: RETRIEVAL_FEATURE_VERSION,
			embeddingModel: cohort.embeddingModel,
			rerankerModel: cohort.rerankerModel,
			scoringVersion: cohort.scoringVersion,
			trainingExamples: scaledExamples.length,
			validationExamples: evaluation.evaluatedExamples,
			trainingPairs: finalTrainingPairs.length,
			validationPairs: evaluation.evaluatedPairs,
			distinctQueries: cohort.distinctQueries,
			datasetStats: dataset.stats,
			scaler: finalScaler,
			ranker,
			evaluation,
			regularization: TRAINING_L2_REGULARIZATION
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unknown retrieval training failure.';

		try {
			await RetrievalModelsRepository.failTrainingRun(run.id, message);
		} catch (persistenceError) {
			console.error(`Failed to mark retrieval training run ${run.id} as failed.`, persistenceError);
		}

		throw error;
	}
}
