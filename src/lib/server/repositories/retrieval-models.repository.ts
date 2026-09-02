import { randomUUID } from 'node:crypto';
import { and, eq } from 'drizzle-orm';
import { db } from '$lib/server/database/database';
import { retrievalRankerModels, retrievalTrainingRuns } from '$lib/server/database/schema';
import type {
	RetrievalTrainingEvaluation,
	RetrievalTrainingHyperparameters
} from '$lib/server/rag/training/retrieval-model.types';
import type { RetrievalFeatureScaler } from '$lib/server/rag/training/retrieval-feature-scaler';
import type { TrainedLinearRetrievalRanker } from '$lib/server/rag/training/train-linear-retrieval-ranker';

interface StartRetrievalTrainingRunInput {
	datasetVersion: number;
	featureVersion: number;
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	trainingExamples: number;
	validationExamples: number;
	distinctQueries: number;
	totalFeedback: number;
	attributedFeedback: number;
	unattributedFeedback: number;
	inconsistentFeedback: number;
	hyperparameters: RetrievalTrainingHyperparameters;
}

interface CompleteRetrievalTrainingRunInput {
	featureVersion: number;
	featureNames: readonly string[];
	scaler: RetrievalFeatureScaler;
	ranker: TrainedLinearRetrievalRanker;
	evaluation: RetrievalTrainingEvaluation;
	regularization: number;
}

export class RetrievalModelsRepository {
	static async startTrainingRun(input: StartRetrievalTrainingRunInput) {
		const [run] = await db
			.insert(retrievalTrainingRuns)
			.values({
				id: randomUUID(),
				status: 'training',
				datasetVersion: input.datasetVersion,
				featureVersion: input.featureVersion,
				embeddingModel: input.embeddingModel,
				rerankerModel: input.rerankerModel,
				scoringVersion: input.scoringVersion,
				trainingExamples: input.trainingExamples,
				validationExamples: input.validationExamples,
				distinctQueries: input.distinctQueries,
				totalFeedback: input.totalFeedback,
				attributedFeedback: input.attributedFeedback,
				unattributedFeedback: input.unattributedFeedback,
				inconsistentFeedback: input.inconsistentFeedback,
				hyperparameters: input.hyperparameters,
				evaluation: null,
				error: null,
				startedAt: new Date().toISOString(),
				completedAt: null
			})
			.returning();

		if (!run) {
			throw new Error('Failed to create the retrieval training run.');
		}

		return run;
	}

	static async completeTrainingRun(runId: string, input: CompleteRetrievalTrainingRunInput) {
		const completedAt = new Date().toISOString();
		const modelId = randomUUID();

		return db.transaction(async (transaction) => {
			const [run] = await transaction
				.update(retrievalTrainingRuns)
				.set({
					status: 'completed',
					evaluation: input.evaluation,
					error: null,
					completedAt
				})
				.where(
					and(eq(retrievalTrainingRuns.id, runId), eq(retrievalTrainingRuns.status, 'training'))
				)
				.returning();

			if (!run) {
				throw new Error(`Training run ${runId} is not in the training state.`);
			}

			const [model] = await transaction
				.insert(retrievalRankerModels)
				.values({
					id: modelId,
					trainingRunId: runId,
					featureVersion: input.featureVersion,
					featureNames: [...input.featureNames],
					means: [...input.scaler.means],
					standardDeviations: [...input.scaler.standardDeviations],
					weights: [...input.ranker.weights],
					intercept: input.ranker.intercept,
					regularization: input.regularization,
					epochs: input.ranker.epochs,
					trainingLoss: input.ranker.trainingLoss,
					createdAt: completedAt
				})
				.returning();

			if (!model) {
				throw new Error(`Failed to save the model for training run ${runId}.`);
			}

			return {
				run,
				model
			};
		});
	}

	static async failTrainingRun(runId: string, message: string): Promise<void> {
		await db
			.update(retrievalTrainingRuns)
			.set({
				status: 'failed',
				error: message.slice(0, 4_000),
				completedAt: new Date().toISOString()
			})
			.where(
				and(eq(retrievalTrainingRuns.id, runId), eq(retrievalTrainingRuns.status, 'training'))
			);
	}

	static findModelById(modelId: string) {
		return db
			.select()
			.from(retrievalRankerModels)
			.where(eq(retrievalRankerModels.id, modelId))
			.get();
	}

	static async findModelWithRunById(modelId: string) {
		const model = await this.findModelById(modelId);
		if (!model) return null;

		const run = await db
			.select()
			.from(retrievalTrainingRuns)
			.where(eq(retrievalTrainingRuns.id, model.trainingRunId))
			.get();

		if (!run) return null;

		return {
			model,
			run
		};
	}
}
