import { randomUUID } from 'node:crypto';
import { and, eq, inArray } from 'drizzle-orm';
import {
	AI_PROXY_FEEDBACK_SOURCE,
	HUMAN_EXPERT_FEEDBACK_SOURCE,
	type RetrievalFeedbackSource
} from '$lib/constants';
import { db } from '$lib/server/database/database';
import {
	retrievalFeedback,
	retrievalImpressionResults,
	retrievalImpressions
} from '$lib/server/database/schema';
import { hashRetrievalQuery } from '$lib/server/rag/search/retrieval-query';
import type { ChunkRatingValue } from '$lib/types';

interface SetRetrievalFeedbackInput {
	chunkId: string;
	impressionResultId: string;
	query: string;
	rating: ChunkRatingValue;
}

interface SetAiProxyRetrievalFeedbackInput extends SetRetrievalFeedbackInput {
	confidence: number;
	rationale: string;
}

interface PersistRetrievalFeedbackInput extends SetRetrievalFeedbackInput {
	feedbackSource: RetrievalFeedbackSource;
	confidence: number | null;
	rationale: string | null;
}

export class RetrievalFeedbackRepository {
	static async findRatings(
		query: string,
		chunkIds: readonly string[]
	): Promise<Map<string, ChunkRatingValue>> {
		const uniqueChunkIds = [...new Set(chunkIds)];
		if (uniqueChunkIds.length === 0) return new Map();

		const rows = await db
			.select({
				chunkId: retrievalFeedback.chunkId,
				rating: retrievalFeedback.rating
			})
			.from(retrievalFeedback)
			.where(
				and(
					eq(retrievalFeedback.queryHash, hashRetrievalQuery(query)),
					eq(retrievalFeedback.feedbackSource, HUMAN_EXPERT_FEEDBACK_SOURCE),
					inArray(retrievalFeedback.chunkId, uniqueChunkIds)
				)
			);

		return new Map(rows.map(({ chunkId, rating }) => [chunkId, rating as ChunkRatingValue]));
	}

	static setHuman(input: SetRetrievalFeedbackInput) {
		return this.set({
			...input,
			feedbackSource: HUMAN_EXPERT_FEEDBACK_SOURCE,
			confidence: null,
			rationale: null
		});
	}

	static setAiProxy(input: SetAiProxyRetrievalFeedbackInput) {
		return this.set({
			...input,
			feedbackSource: AI_PROXY_FEEDBACK_SOURCE,
			confidence: input.confidence,
			rationale: input.rationale
		});
	}

	private static async set(input: PersistRetrievalFeedbackInput) {
		const query = input.query.trim();
		const queryHash = hashRetrievalQuery(query);
		const attribution = await db
			.select({
				retrievalMode: retrievalImpressionResults.retrievalMode,
				resultRank: retrievalImpressionResults.displayedRank
			})
			.from(retrievalImpressionResults)
			.innerJoin(
				retrievalImpressions,
				eq(retrievalImpressions.id, retrievalImpressionResults.impressionId)
			)
			.where(
				and(
					eq(retrievalImpressionResults.id, input.impressionResultId),
					eq(retrievalImpressionResults.chunkId, input.chunkId),
					eq(retrievalImpressions.queryHash, queryHash)
				)
			)
			.get();

		if (!attribution) return null;

		const timestamp = new Date().toISOString();
		const [row] = await db
			.insert(retrievalFeedback)
			.values({
				id: randomUUID(),
				chunkId: input.chunkId,
				impressionResultId: input.impressionResultId,
				query,
				queryHash,
				rating: input.rating,
				feedbackSource: input.feedbackSource,
				confidence: input.confidence,
				rationale: input.rationale,
				retrievalMode: attribution.retrievalMode,
				resultRank: attribution.resultRank,
				createdAt: timestamp,
				updatedAt: timestamp
			})
			.onConflictDoUpdate({
				target: [
					retrievalFeedback.chunkId,
					retrievalFeedback.queryHash,
					retrievalFeedback.feedbackSource
				],
				set: {
					impressionResultId: input.impressionResultId,
					query,
					rating: input.rating,
					confidence: input.confidence,
					rationale: input.rationale,
					retrievalMode: attribution.retrievalMode,
					resultRank: attribution.resultRank,
					updatedAt: timestamp
				}
			})
			.returning();

		return row;
	}

	static clearHuman(chunkId: string, query: string): Promise<void> {
		return this.clear(chunkId, query, HUMAN_EXPERT_FEEDBACK_SOURCE);
	}

	static clearAiProxy(chunkId: string, query: string): Promise<void> {
		return this.clear(chunkId, query, AI_PROXY_FEEDBACK_SOURCE);
	}

	private static async clear(
		chunkId: string,
		query: string,
		feedbackSource: RetrievalFeedbackSource
	): Promise<void> {
		await db
			.delete(retrievalFeedback)
			.where(
				and(
					eq(retrievalFeedback.chunkId, chunkId),
					eq(retrievalFeedback.queryHash, hashRetrievalQuery(query)),
					eq(retrievalFeedback.feedbackSource, feedbackSource)
				)
			);
	}
}
