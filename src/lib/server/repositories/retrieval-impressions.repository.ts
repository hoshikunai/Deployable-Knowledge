import { randomUUID } from 'node:crypto';
import { db } from '$lib/server/database/database';
import { retrievalImpressionResults, retrievalImpressions } from '$lib/server/database/schema';
import { hashRetrievalQuery } from '$lib/server/rag/search/retrieval-query';
import type { RetrievalCandidateSnapshot } from '$lib/server/rag/search/search-shared';

interface RecordRetrievalImpressionInput {
	query: string;
	requestedTopK: number;
	documentIds: string[];
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	candidates: RetrievalCandidateSnapshot[];
}

export class RetrievalImpressionsRepository {
	static async record(input: RecordRetrievalImpressionInput): Promise<string> {
		const impressionId = randomUUID();
		const createdAt = new Date().toISOString();

		await db.transaction(async (transaction) => {
			await transaction.insert(retrievalImpressions).values({
				id: impressionId,
				query: input.query.trim(),
				queryHash: hashRetrievalQuery(input.query),
				requestedTopK: input.requestedTopK,
				documentIds: [...input.documentIds],
				embeddingModel: input.embeddingModel,
				rerankerModel: input.rerankerModel,
				scoringVersion: input.scoringVersion,
				createdAt
			});

			if (input.candidates.length === 0) return;

			await transaction.insert(retrievalImpressionResults).values(
				input.candidates.map((candidate) => ({
					id: randomUUID(),
					impressionId,
					...candidate
				}))
			);
		});

		return impressionId;
	}
}
