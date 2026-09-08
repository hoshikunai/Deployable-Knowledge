import { desc, eq, inArray } from 'drizzle-orm';
import type { RetrievalFeedbackSource } from '$lib/constants';
import { db } from '$lib/server/database/database';
import {
	retrievalFeedback,
	retrievalImpressionResults,
	retrievalImpressions
} from '$lib/server/database/schema';

export class RetrievalTrainingRepository {
	static async readCandidateRows(impressionIds: string[]) {
		if (impressionIds.length === 0) return [];

		return db
			.select({
				impressionResultId: retrievalImpressionResults.id,
				impressionId: retrievalImpressionResults.impressionId,
				retrievalMode: retrievalImpressionResults.retrievalMode,
				baseRank: retrievalImpressionResults.baseRank,
				semanticScore: retrievalImpressionResults.semanticScore,
				bm25Score: retrievalImpressionResults.bm25Score,
				crossEncoderScore: retrievalImpressionResults.crossEncoderScore,
				baseScore: retrievalImpressionResults.baseScore
			})
			.from(retrievalImpressionResults)
			.where(inArray(retrievalImpressionResults.impressionId, impressionIds));
	}

	static readDatasetRows(feedbackSource: RetrievalFeedbackSource) {
		return db
			.select({
				feedbackId: retrievalFeedback.id,
				feedbackChunkId: retrievalFeedback.chunkId,
				feedbackImpressionResultId: retrievalFeedback.impressionResultId,
				queryHash: retrievalFeedback.queryHash,
				rating: retrievalFeedback.rating,
				feedbackRetrievalMode: retrievalFeedback.retrievalMode,
				feedbackResultRank: retrievalFeedback.resultRank,
				feedbackSource: retrievalFeedback.feedbackSource,
				feedbackConfidence: retrievalFeedback.confidence,
				feedbackRationale: retrievalFeedback.rationale,
				feedbackUpdatedAt: retrievalFeedback.updatedAt,
				resultId: retrievalImpressionResults.id,
				resultChunkId: retrievalImpressionResults.chunkId,
				resultRetrievalMode: retrievalImpressionResults.retrievalMode,
				baseRank: retrievalImpressionResults.baseRank,
				displayedRank: retrievalImpressionResults.displayedRank,
				semanticScore: retrievalImpressionResults.semanticScore,
				bm25Score: retrievalImpressionResults.bm25Score,
				crossEncoderScore: retrievalImpressionResults.crossEncoderScore,
				baseScore: retrievalImpressionResults.baseScore,
				impressionId: retrievalImpressions.id,
				requestedTopK: retrievalImpressions.requestedTopK,
				embeddingModel: retrievalImpressions.embeddingModel,
				rerankerModel: retrievalImpressions.rerankerModel,
				scoringVersion: retrievalImpressions.scoringVersion,
				impressionCreatedAt: retrievalImpressions.createdAt
			})
			.from(retrievalFeedback)
			.leftJoin(
				retrievalImpressionResults,
				eq(retrievalFeedback.impressionResultId, retrievalImpressionResults.id)
			)
			.leftJoin(
				retrievalImpressions,
				eq(retrievalImpressionResults.impressionId, retrievalImpressions.id)
			)
			.where(eq(retrievalFeedback.feedbackSource, feedbackSource))
			.orderBy(desc(retrievalFeedback.updatedAt));
	}
}
