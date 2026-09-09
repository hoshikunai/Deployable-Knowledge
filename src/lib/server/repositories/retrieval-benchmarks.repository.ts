import { randomUUID } from 'node:crypto';
import { asc, inArray } from 'drizzle-orm';
import { db } from '$lib/server/database/database';
import {
	documentChunks,
	documents,
	retrievalBenchmarkCases,
	retrievalBenchmarkJudgments
} from '$lib/server/database/schema';
import type {
	CreateRetrievalBenchmarkCaseInput,
	RetrievalBenchmarkCase,
	RetrievalBenchmarkJudgment
} from '$lib/server/rag/benchmark/retrieval-benchmark.types';

export class RetrievalBenchmarksRepository {
	static async create(input: CreateRetrievalBenchmarkCaseInput): Promise<RetrievalBenchmarkCase> {
		const id = randomUUID();
		const createdAt = new Date().toISOString();

		await db.transaction(async (transaction) => {
			if (input.documentIds.length > 0) {
				const selectedDocuments = await transaction
					.select({ id: documents.id })
					.from(documents)
					.where(inArray(documents.id, input.documentIds));
				if (selectedDocuments.length !== new Set(input.documentIds).size) {
					throw new Error('A benchmark document filter references a missing document.');
				}
			}

			const judgedChunks = await transaction
				.select({ id: documentChunks.id, documentId: documentChunks.documentId })
				.from(documentChunks)
				.where(
					inArray(
						documentChunks.id,
						input.judgments.map(({ chunkId }) => chunkId)
					)
				);
			if (
				judgedChunks.length !== input.judgments.length ||
				(input.documentIds.length > 0 &&
					judgedChunks.some((chunk) => !input.documentIds.includes(chunk.documentId)))
			) {
				throw new Error(
					'Benchmark judgments must reference existing chunks within the document filters.'
				);
			}

			await transaction.insert(retrievalBenchmarkCases).values({
				id,
				name: input.name,
				query: input.query,
				documentIds: input.documentIds,
				createdAt
			});

			await transaction.insert(retrievalBenchmarkJudgments).values(
				input.judgments.map((judgment) => ({
					caseId: id,
					chunkId: judgment.chunkId,
					relevance: judgment.relevance,
					createdAt
				}))
			);
		});

		return {
			id,
			name: input.name,
			query: input.query,
			documentIds: input.documentIds,
			judgments: input.judgments,
			createdAt
		};
	}

	static async list(): Promise<RetrievalBenchmarkCase[]> {
		const [caseRows, judgmentRows] = await Promise.all([
			db.select().from(retrievalBenchmarkCases).orderBy(asc(retrievalBenchmarkCases.createdAt)),
			db
				.select()
				.from(retrievalBenchmarkJudgments)
				.orderBy(asc(retrievalBenchmarkJudgments.createdAt))
		]);

		const judgmentsByCase = new Map<string, RetrievalBenchmarkJudgment[]>();

		for (const row of judgmentRows) {
			const judgments = judgmentsByCase.get(row.caseId) ?? [];
			judgments.push({
				chunkId: row.chunkId,
				relevance: row.relevance as RetrievalBenchmarkJudgment['relevance']
			});
			judgmentsByCase.set(row.caseId, judgments);
		}

		return caseRows.map((row) => ({
			id: row.id,
			name: row.name,
			query: row.query,
			documentIds: row.documentIds,
			judgments: judgmentsByCase.get(row.id) ?? [],
			createdAt: row.createdAt
		}));
	}
}
