import { randomUUID } from 'node:crypto';
import { and, eq, inArray } from 'drizzle-orm';
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
					inArray(retrievalFeedback.chunkId, uniqueChunkIds)
				)
			);

		return new Map(rows.map(({ chunkId, rating }) => [chunkId, rating as ChunkRatingValue]));
	}

	static async set(input: SetRetrievalFeedbackInput) {
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
				retrievalMode: attribution.retrievalMode,
				resultRank: attribution.resultRank,
				createdAt: timestamp,
				updatedAt: timestamp
			})
			.onConflictDoUpdate({
				target: [retrievalFeedback.chunkId, retrievalFeedback.queryHash],
				set: {
					impressionResultId: input.impressionResultId,
					query,
					rating: input.rating,
					retrievalMode: attribution.retrievalMode,
					resultRank: attribution.resultRank,
					updatedAt: timestamp
				}
			})
			.returning();

		return row;
	}

	static async clear(chunkId: string, query: string): Promise<void> {
		await db
			.delete(retrievalFeedback)
			.where(
				and(
					eq(retrievalFeedback.chunkId, chunkId),
					eq(retrievalFeedback.queryHash, hashRetrievalQuery(query))
				)
			);
	}
}
