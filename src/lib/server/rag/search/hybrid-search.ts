// Hybrid search gathers semantic and BM25 candidates and fuses them with RRF.
// When a reranker is configured, it reorders a bounded RRF shortlist.

import { searchSemantic } from './semantic-search';
import { searchBm25 } from './bm25-search';
import { rerankCandidates } from './cross-rerank';
import { requireCrossEncoder } from './cross-encoders/registry';
import { hybridPipeline, type HybridPipelineConfig } from './hybrid-pipeline';
import { fuseSearchResultsRrf, type RrfFusedMatch } from './rrf-fusion';
import { type ScoredSearchMatch, type SearchOptionsBase, type SearchResult } from './search-shared';

type SearchMethodResults = {
	query: string;
	semantic: ScoredSearchMatch[];
	bm25: ScoredSearchMatch[];
	hybrid: ScoredSearchMatch[];
	pipeline: HybridPipelineConfig;
};

async function rerankShortlist(
	query: string,
	fusedCandidates: RrfFusedMatch[],
	topK: number,
	rerankerId: string
): Promise<ScoredSearchMatch[]> {
	const shortlist = fusedCandidates.slice(0, topK * hybridPipeline.rerankMultiplier);
	const matchesByChunkId = new Map(shortlist.map(({ match }) => [match.chunkId, match]));
	const crossEncoder = requireCrossEncoder(rerankerId);

	const rerankedCandidates = await rerankCandidates(
		query,
		shortlist.map(({ match }) => ({
			chunkId: match.chunkId,
			content: match.content
		})),
		crossEncoder,
		hybridPipeline.rerankMaxTokens
	);

	const reranked: ScoredSearchMatch[] = [];

	for (const candidate of rerankedCandidates) {
		const match = matchesByChunkId.get(candidate.chunkId);

		if (!match) {
			throw new Error(`${crossEncoder.name} returned an unknown chunk ID: ${candidate.chunkId}`);
		}

		reranked.push({
			...match,
			score: candidate.score
		});

		if (reranked.length === topK) {
			break;
		}
	}

	return reranked;
}

async function collectMethodResults(options: SearchOptionsBase): Promise<{
	query: string;
	semantic: ScoredSearchMatch[];
	bm25: ScoredSearchMatch[];
	hybridScored: ScoredSearchMatch[];
}> {
	const query = options.query.trim();
	const topK = Math.max(0, Math.floor(options.topK ?? 10));

	if (!query || topK === 0) {
		return {
			query,
			semantic: [],
			bm25: [],
			hybridScored: []
		};
	}

	const sharedOptions = {
		...options,
		query,
		topK: topK * hybridPipeline.retrievalMultiplier
	};

	const [semanticSearch, bm25Search] = await Promise.all([
		searchSemantic(sharedOptions),
		searchBm25(sharedOptions)
	]);

	const fusedCandidates = fuseSearchResultsRrf(semanticSearch.results, bm25Search.results, {
		rankConstant: hybridPipeline.rrfRankConstant
	});

	const hybridScored = hybridPipeline.reranker
		? await rerankShortlist(query, fusedCandidates, topK, hybridPipeline.reranker)
		: fusedCandidates.slice(0, topK).map(({ match, score }) => ({ ...match, score }));

	return {
		query,
		semantic: semanticSearch.results.slice(0, topK),
		bm25: bm25Search.results.slice(0, topK),
		hybridScored
	};
}

export async function searchAllMethods(options: SearchOptionsBase): Promise<SearchMethodResults> {
	const search = await collectMethodResults(options);

	return {
		query: search.query,
		semantic: search.semantic,
		bm25: search.bm25,
		hybrid: search.hybridScored,
		pipeline: hybridPipeline
	};
}

export async function searchHybrid(
	options: SearchOptionsBase
): Promise<SearchResult<ScoredSearchMatch>> {
	const search = await collectMethodResults(options);

	return {
		query: search.query,
		results: search.hybridScored
	};
}
