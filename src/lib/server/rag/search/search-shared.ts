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
	displayedRank: number;
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
	const baseResultsByChunk = new Map(
		baseResults.map((match, index) => [
			match.chunkId,
			{
				rank: index + 1,
				score: match.score
			}
		])
	);

	return displayedResults.map((match, index) => {
		const baseResult = baseResultsByChunk.get(match.chunkId);
		if (!baseResult) {
			throw new Error(`Displayed chunk ${match.chunkId} is missing from the base ranking.`);
		}

		return {
			chunkId: match.chunkId,
			retrievalMode,
			baseRank: baseResult.rank,
			displayedRank: index + 1,
			semanticScore: scores.semantic.get(match.chunkId) ?? null,
			bm25Score: scores.bm25.get(match.chunkId) ?? null,
			crossEncoderScore: scores.crossEncoder.get(match.chunkId) ?? null,
			baseScore: baseResult.score,
			learnedScore: learnedScores.get(match.chunkId) ?? null
		};
	});
}
