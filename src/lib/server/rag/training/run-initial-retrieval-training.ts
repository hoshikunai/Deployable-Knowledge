import { RetrievalModelsRepository } from '$lib/server/repositories';
import { buildRetrievalTrainingDataset } from './build-retrieval-training-dataset';
import {
	buildRetrievalTrainingFeatures,
	RETRIEVAL_FEATURE_NAMES
} from './build-retrieval-training-features';
import {
	evaluateRetrievalRanker,
	type RetrievalTrainingEvaluation
} from './evaluate-retrieval-ranker';
import {
	fitRetrievalFeatureScaler,
	scaleRetrievalTrainingExamples,
	type RetrievalFeatureScaler
} from './retrieval-feature-scaler';
import type { RetrievalTrainingHyperparameters } from './retrieval-model.types';
import {
	RETRIEVAL_FEATURE_VERSION,
	TRAINING_EARLY_STOPPING_PATIENCE,
	TRAINING_L2_REGULARIZATION,
	TRAINING_LEARNING_RATE,
	TRAINING_LOSS_TOLERANCE,
	TRAINING_MAX_EPOCHS,
	RETRIEVAL_RATING_BALANCE_STRATEGY,
	VALIDATION_FRACTION
} from './retrieval-training-constants';
import type { RetrievalTrainingDatasetStats } from './retrieval-training.types';
import {
	selectRetrievalTrainingCohort,
	type RetrievalTrainingCohort
} from './select-retrieval-training-cohort';
import { splitRetrievalTrainingDataset } from './split-retrieval-training-dataset';
import {
	trainLinearRetrievalRanker,
	type TrainedLinearRetrievalRanker
} from './train-linear-retrieval-ranker';

export interface InitialRetrievalTrainingResult {
	runId: string;
	modelId: string;
	featureVersion: typeof RETRIEVAL_FEATURE_VERSION;
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	trainingExamples: number;
	validationExamples: number;
	distinctQueries: number;
	datasetStats: RetrievalTrainingDatasetStats;
	scaler: RetrievalFeatureScaler;
	ranker: TrainedLinearRetrievalRanker;
	evaluation: RetrievalTrainingEvaluation;
	regularization: number;
}

export async function runInitialRetrievalTraining(): Promise<InitialRetrievalTrainingResult> {
	const dataset = await buildRetrievalTrainingDataset();
	const cohort: RetrievalTrainingCohort = selectRetrievalTrainingCohort(dataset.examples);
	const preparedExamples = buildRetrievalTrainingFeatures(cohort.examples);
	const split = splitRetrievalTrainingDataset(preparedExamples);

	const hyperparameters: RetrievalTrainingHyperparameters = {
		learningRate: TRAINING_LEARNING_RATE,
		l2Regularization: TRAINING_L2_REGULARIZATION,
		maxEpochs: TRAINING_MAX_EPOCHS,
		earlyStoppingPatience: TRAINING_EARLY_STOPPING_PATIENCE,
		lossTolerance: TRAINING_LOSS_TOLERANCE,
		validationFraction: VALIDATION_FRACTION,
		ratingBalanceStrategy: RETRIEVAL_RATING_BALANCE_STRATEGY
	};

	const run = await RetrievalModelsRepository.startTrainingRun({
		datasetVersion: dataset.version,
		featureVersion: RETRIEVAL_FEATURE_VERSION,
		embeddingModel: cohort.embeddingModel,
		rerankerModel: cohort.rerankerModel,
		scoringVersion: cohort.scoringVersion,
		trainingExamples: split.training.length,
		validationExamples: split.validation.length,
		distinctQueries: cohort.distinctQueries,
		totalFeedback: dataset.stats.totalFeedback,
		attributedFeedback: dataset.stats.attributedFeedback,
		unattributedFeedback: dataset.stats.unattributedFeedback,
		inconsistentFeedback: dataset.stats.inconsistentFeedback,
		hyperparameters
	});

	try {
		const scaler = fitRetrievalFeatureScaler(split.training);
		const scaledTraining = scaleRetrievalTrainingExamples(split.training, scaler);
		const scaledValidation = scaleRetrievalTrainingExamples(split.validation, scaler);

		const ranker = trainLinearRetrievalRanker(scaledTraining);
		const evaluation = evaluateRetrievalRanker(ranker, scaledValidation);

		const saved = await RetrievalModelsRepository.completeTrainingRun(run.id, {
			featureVersion: RETRIEVAL_FEATURE_VERSION,
			featureNames: RETRIEVAL_FEATURE_NAMES,
			scaler,
			ranker,
			evaluation,
			regularization: TRAINING_L2_REGULARIZATION
		});

		return {
			runId: saved.run.id,
			modelId: saved.model.id,
			featureVersion: RETRIEVAL_FEATURE_VERSION,
			embeddingModel: cohort.embeddingModel,
			rerankerModel: cohort.rerankerModel,
			scoringVersion: cohort.scoringVersion,
			trainingExamples: scaledTraining.length,
			validationExamples: scaledValidation.length,
			distinctQueries: cohort.distinctQueries,
			datasetStats: dataset.stats,
			scaler,
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
