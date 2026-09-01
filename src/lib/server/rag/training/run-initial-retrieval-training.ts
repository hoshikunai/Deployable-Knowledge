import { buildRetrievalTrainingDataset } from './build-retrieval-training-dataset';
import { buildRetrievalTrainingFeatures } from './build-retrieval-training-features';
import {
	evaluateRetrievalRanker,
	type RetrievalTrainingEvaluation
} from './evaluate-retrieval-ranker';
import {
	fitRetrievalFeatureScaler,
	scaleRetrievalTrainingExamples,
	type RetrievalFeatureScaler
} from './retrieval-feature-scaler';
import {
	RETRIEVAL_FEATURE_VERSION,
	TRAINING_L2_REGULARIZATION
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

	const scaler = fitRetrievalFeatureScaler(split.training);
	const scaledTraining = scaleRetrievalTrainingExamples(split.training, scaler);
	const scaledValidation = scaleRetrievalTrainingExamples(split.validation, scaler);

	const ranker = trainLinearRetrievalRanker(scaledTraining);
	const evaluation = evaluateRetrievalRanker(ranker, scaledValidation);

	return {
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
}
