// Shared helpers for search modules

import type { Document, DocumentChunk } from '../../database/schema';
import type { RetrievalMode } from '$lib/enums';

// Same chunk type values used by stored chunks and every search mode
export type SearchChunkType = DocumentChunk['chunkType'];

// Common search input shared by BM25, semantic, and hybrid search
export type SearchOptionsBase = {
	query: string;
	topK?: number;
	documentIds?: string[];
	sourcePaths?: string[];
	chunkTypes?: SearchChunkType[];
};

// Common fields returned by every chunk search result
export type SearchMatchBase = {
	chunkId: string;
	documentId: string;
	sourcePath: string;
	sourceTitle: string;
	sourceType: Document['sourceType'];
	pageIndex: number;
	chunkIndex: number;
	chunkType: SearchChunkType;
	content: string;
};

export type ScoredSearchMatch = SearchMatchBase & {
	score: number;
};

export interface RetrievalCandidateSnapshot {
	chunkId: string;
	retrievalMode: RetrievalMode;
	baseRank: number;
	displayedRank: number | null;
	wasDisplayed: boolean;
	semanticScore: number | null;
	bm25Score: number | null;
	crossEncoderScore: number | null;
	baseScore: number;
	learnedScore: number | null;
}

export interface RetrievalScoreMaps {
	semantic: ReadonlyMap<string, number>;
	bm25: ReadonlyMap<string, number>;
	crossEncoder: ReadonlyMap<string, number>;
}

export type RelevanceSearchMatch = SearchMatchBase & {
	relevanceScore: number;
};

// All search modules return the normalized query plus ranked results
export type SearchResult<TMatch extends SearchMatchBase> = {
	query: string;
	results: TMatch[];
};

// Search filters come from user/UI input, so trim, drop blanks, and dedupe before querying
export function cleanFilterValues<T extends string>(values: readonly T[] | undefined): T[] {
	const cleaned = new Set<T>();
	for (const value of values ?? []) {
		const trimmed = value.trim();
		if (trimmed) {
			cleaned.add(trimmed as T);
		}
	}

	return [...cleaned];
}

export function buildRetrievalCandidateSnapshots(
	retrievalMode: RetrievalMode,
	baseResults: ScoredSearchMatch[],
	displayedResults: ScoredSearchMatch[],
	scores: RetrievalScoreMaps,
	learnedScores: ReadonlyMap<string, number> = new Map()
): RetrievalCandidateSnapshot[] {
	const baseChunkIds = new Set(baseResults.map(({ chunkId }) => chunkId));
	const displayedRanks = new Map(
		displayedResults.map(({ chunkId }, index) => [chunkId, index + 1])
	);

	for (const match of displayedResults) {
		if (!baseChunkIds.has(match.chunkId)) {
			throw new Error(`Displayed chunk ${match.chunkId} is missing from the candidate pool.`);
		}
	}

	return baseResults.map((match, index) => {
		const displayedRank = displayedRanks.get(match.chunkId) ?? null;

		return {
			chunkId: match.chunkId,
			retrievalMode,
			baseRank: index + 1,
			displayedRank,
			wasDisplayed: displayedRank !== null,
			semanticScore: scores.semantic.get(match.chunkId) ?? null,
			bm25Score: scores.bm25.get(match.chunkId) ?? null,
			crossEncoderScore: scores.crossEncoder.get(match.chunkId) ?? null,
			baseScore: match.score,
			learnedScore: learnedScores.get(match.chunkId) ?? null
		};
	});
}
