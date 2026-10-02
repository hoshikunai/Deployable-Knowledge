// Hybrid search gathers semantic and BM25 candidates and fuses them with RRF (by chunk or by
// document, with optional retriever weights). Optional (experimental) dedup trims repeats, a
// configured reranker reorders a bounded RRF shortlist, and optional document ordering groups
// the final results by document.

import { searchSemantic } from './semantic-search';
import { searchBm25 } from './bm25-search';
import { rerankCandidates } from './cross-rerank';
import { requireCrossEncoder } from './cross-encoders/registry';
import { hybridPipeline, type HybridPipelineConfig } from './hybrid-pipeline';
import { activeEmbeddingModel } from '../embedding-models/registry';
import { dedupeCandidates } from './candidate-dedup';
import { orderByDocument } from './document-scoring';
import { fuseSearchResultsRrf, type RrfFusedMatch } from './rrf-fusion';
import { type ScoredSearchMatch, type SearchOptionsBase, type SearchResult } from './search-shared';

type SearchMethodResults = {
	query: string;
	semantic: ScoredSearchMatch[];
	bm25: ScoredSearchMatch[];
	hybrid: ScoredSearchMatch[];
	pipeline: HybridPipelineConfig;
	// Reported so benchmarks can confirm which model embedded the query
	embeddingModel: string;
};

function takeCandidates(fusedCandidates: RrfFusedMatch[], limit: number): RrfFusedMatch[] {
	return dedupeCandidates(
		fusedCandidates,
		hybridPipeline.dedup,
		limit,
		hybridPipeline.maxChunksPerDocument
	);
}

function rerankInput(match: ScoredSearchMatch): string {
	const rawTitle = match.sourceTitle.trim();
	const title = rawTitle
		.replace(/\.[a-z][a-z0-9]{1,4}$/i, '')
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();
	const hasWord = /\p{L}{4,}/u.test(title);
	if (!hybridPipeline.rerankTitle || !hasWord || match.content.startsWith(rawTitle)) {
		return match.content;
	}
	return `${title}\n\n${match.content}`;
}

// Returns the whole shortlist in reranked order; callers cut it to topK
async function rerankShortlist(
	query: string,
	fusedCandidates: RrfFusedMatch[],
	topK: number,
	rerankerId: string
): Promise<ScoredSearchMatch[]> {
	const shortlist = takeCandidates(fusedCandidates, topK * hybridPipeline.rerankMultiplier);
	const matchesByChunkId = new Map(shortlist.map(({ match }) => [match.chunkId, match]));
	const crossEncoder = requireCrossEncoder(rerankerId);

	const rerankedCandidates = await rerankCandidates(
		query,
		shortlist.map(({ match }) => ({
			chunkId: match.chunkId,
			content: rerankInput(match)
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
		rankConstant: hybridPipeline.rrfRankConstant,
		semanticWeight: hybridPipeline.rrfSemanticWeight,
		bm25Weight: hybridPipeline.rrfBm25Weight,
		level: hybridPipeline.fusionLevel
	});

	// Document ordering looks past the first topK chunks so it can see each document's
	// other chunks; without it, only the first topK are needed.
	const poolSize =
		hybridPipeline.documentOrder === 'none' ? topK : topK * hybridPipeline.rerankMultiplier;
	const ranked = hybridPipeline.reranker
		? await rerankShortlist(query, fusedCandidates, topK, hybridPipeline.reranker)
		: takeCandidates(fusedCandidates, poolSize).map(({ match, score }) => ({ ...match, score }));
	const hybridScored = orderByDocument(ranked, hybridPipeline.documentOrder, topK);

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
		pipeline: hybridPipeline,
		embeddingModel: activeEmbeddingModel.id
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
