// Exact semantic search over the stored chunk embeddings in SQLite

import { and, eq, inArray } from 'drizzle-orm';
import type { SQL } from 'drizzle-orm';
import { DEFAULT_ASSISTANT_CONFIG } from '$lib/constants';
import { db } from '../../database/database';
import { documentChunks, documents } from '../../database/schema';
import { embedTexts } from '../embedding-model';
import { embeddingDotProduct, embeddingFromBytes } from '../embedding-vectors';
import {
	cleanFilterValues,
	type ScoredSearchMatch,
	type SearchChunkType,
	type SearchMatchBase,
	type SearchOptionsBase,
	type SearchResult
} from './search-shared';

export type SemanticSearchMatch = ScoredSearchMatch;
export type SemanticSearchResult = SearchResult<SemanticSearchMatch>;

type CandidateRow = {
	chunkId: string;
	documentId: string;
	sourcePath: string;
	sourceTitle: string;
	sourceType: SearchMatchBase['sourceType'];
	pageIndex: number;
	chunkIndex: number;
	chunkType: SearchChunkType;
	content: string;
	embedding: Uint8Array | ArrayBuffer | null;
};

// Semantic search scores the query against stored chunk embeddings already in SQLite
export async function searchSemantic(options: SearchOptionsBase): Promise<SemanticSearchResult> {
	const query = options.query.trim();
	// Keeps topK as a non-negative integer before using it as a result limit
	const topK = Math.max(0, Math.floor(options.topK ?? DEFAULT_ASSISTANT_CONFIG.ragTopK));
	const documentIds = cleanFilterValues(options.documentIds);
	const sourcePaths = cleanFilterValues(options.sourcePaths);
	const chunkTypes = cleanFilterValues(options.chunkTypes);

	// Empty queries should not run embedding/model work.
	if (!query || topK === 0) {
		return {
			query,
			results: []
		};
	}

	// Same embedding path as chunking/storage so query vectors stay in sync with the corpus
	const queryEmbedding = (await embedTexts([query], 'search_query'))[0] ?? [];
	// Deactivated documents never surface in retrieval, even when explicitly requested
	const filters: SQL[] = [eq(documents.active, true)];

	if (documentIds.length > 0) {
		filters.push(inArray(documentChunks.documentId, documentIds));
	}

	if (sourcePaths.length > 0) {
		filters.push(inArray(documents.sourcePath, sourcePaths));
	}

	if (chunkTypes.length > 0) {
		filters.push(inArray(documentChunks.chunkType, chunkTypes));
	}

	// Use Drizzle for the row query, then do vector math in TS
	const candidateRows = (await db
		.select({
			chunkId: documentChunks.id,
			documentId: documentChunks.documentId,
			sourcePath: documents.sourcePath,
			sourceTitle: documents.title,
			sourceType: documents.sourceType,
			pageIndex: documentChunks.pageIndex,
			chunkIndex: documentChunks.chunkIndex,
			chunkType: documentChunks.chunkType,
			content: documentChunks.content,
			embedding: documentChunks.embedding
		})
		.from(documentChunks)
		.innerJoin(documents, eq(documents.id, documentChunks.documentId))
		.where(filters.length ? and(...filters) : undefined)) as CandidateRow[];

	// Stored vectors are Float32 bytes. Decode them once before scoring
	const decodedCandidates = candidateRows.map((row) => {
		const rawEmbedding = row.embedding;

		if (!rawEmbedding) {
			throw new Error(`Chunk ${row.chunkId} is missing its embedding bytes.`);
		}

		if (!(rawEmbedding instanceof Uint8Array) && !(rawEmbedding instanceof ArrayBuffer)) {
			throw new Error(`Chunk ${row.chunkId} returned an unsupported embedding shape.`);
		}

		return {
			row,
			vector: embeddingFromBytes(rawEmbedding)
		};
	});

	const scoredRows: SemanticSearchMatch[] = [];

	for (const candidate of decodedCandidates) {
		const { row, vector } = candidate;

		// Embeddings are normalized, so dot product is the cosine score
		const score = embeddingDotProduct(queryEmbedding, vector);
		scoredRows.push({
			chunkId: row.chunkId,
			documentId: row.documentId,
			sourcePath: row.sourcePath,
			sourceTitle: row.sourceTitle,
			sourceType: row.sourceType,
			pageIndex: row.pageIndex,
			chunkIndex: row.chunkIndex,
			chunkType: row.chunkType,
			content: row.content,
			score
		});
	}

	scoredRows.sort((left, right) => right.score - left.score);
	const results = scoredRows.slice(0, topK);

	return {
		query,
		results
	};
}
