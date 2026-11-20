import { searchSemantic } from './semantic-search';
import { searchBm25 } from './bm25-search';
import { rerankCandidates } from './cross-rerank';
import { findCrossEncoder } from './cross-encoders/registry';
import { dedupeCandidates } from './candidate-dedup';
import { orderByDocument } from './document-scoring';
import { getHybridPipeline, type HybridPipelineConfig } from './hybrid-pipeline';
import { fuseSearchResultsRrf } from './rrf-fusion';
import { getActiveEmbedding } from '../embedding-model';
import { pdfPageCleanup } from '../chunk/pdf-config';
import {
	type ScoredSearchMatch,
	type SearchMatchBase,
	type SearchOptionsBase,
	type SearchResult
} from './search-shared';

type SearchMethodResults = {
	query: string;
	semantic: SearchMatchBase[];
	bm25: SearchMatchBase[];
	hybrid: SearchMatchBase[];
	pipeline: HybridPipelineConfig;
	embeddingModel: string;
	pdfPageCleanup: boolean;
};

function withoutScore(match: ScoredSearchMatch): SearchMatchBase {
	const { score: _score, ...chunk } = match;
	return chunk;
}

function unionCandidates(
	semantic: readonly ScoredSearchMatch[],
	bm25: readonly ScoredSearchMatch[]
): ScoredSearchMatch[] {
	const byChunkId = new Map<string, SearchMatchBase>();

	for (const match of [...semantic, ...bm25]) {
		if (!byChunkId.has(match.chunkId)) {
			byChunkId.set(match.chunkId, withoutScore(match));
		}
	}

	return [...byChunkId.values()].map((match, index) => ({ ...match, score: 1 / (index + 1) }));
}

function rrfCandidates(
	semantic: readonly ScoredSearchMatch[],
	bm25: readonly ScoredSearchMatch[],
	topK: number,
	pipeline: HybridPipelineConfig
): ScoredSearchMatch[] {
	const fused = fuseSearchResultsRrf(semantic, bm25, {
		rankConstant: pipeline.rrfRankConstant,
		semanticWeight: pipeline.rrfSemanticWeight,
		bm25Weight: pipeline.rrfBm25Weight,
		level: pipeline.fusionLevel
	});
	const widened = pipeline.reranker !== null || pipeline.documentOrder !== 'none';
	const limit = widened ? topK * pipeline.rerankMultiplier : topK;

	return dedupeCandidates(fused, pipeline.dedup, limit, pipeline.maxChunksPerDocument).map(
		({ match, score }) => ({ ...match, score })
	);
}

function rerankInput(match: SearchMatchBase, includeTitle: boolean): string {
	const rawTitle = match.sourceTitle.trim();
	const title = rawTitle
		.replace(/\.[a-z][a-z0-9]{1,4}$/i, '')
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

	if (!includeTitle || !/\p{L}{4,}/u.test(title) || match.content.startsWith(rawTitle)) {
		return match.content;
	}

	return `${title}\n\n${match.content}`;
}

async function rerankMatches(
	query: string,
	candidates: readonly ScoredSearchMatch[],
	rerankerId: string,
	pipeline: HybridPipelineConfig
): Promise<ScoredSearchMatch[]> {
	const crossEncoder = findCrossEncoder(rerankerId);
	if (!crossEncoder) throw new Error(`Cross-encoder "${rerankerId}" is unavailable.`);

	const byChunkId = new Map(candidates.map((match) => [match.chunkId, match]));
	const rankedCandidates = await rerankCandidates(
		query,
		candidates.map((match) => ({
			chunkId: match.chunkId,
			content: rerankInput(match, pipeline.rerankTitle)
		})),
		crossEncoder,
		pipeline.rerankMaxTokens
	);
	const reranked: ScoredSearchMatch[] = [];

	for (const candidate of rankedCandidates) {
		const match = byChunkId.get(candidate.chunkId);
		if (match) reranked.push({ ...match, score: candidate.relevance });
	}

	return reranked;
}

async function collectMethodResults(options: SearchOptionsBase): Promise<{
	query: string;
	semantic: SearchMatchBase[];
	bm25: SearchMatchBase[];
	hybridScored: ScoredSearchMatch[];
}> {
	const query = options.query.trim();
	const topK = Math.max(0, Math.floor(options.topK ?? 10));

	if (!query || topK === 0) {
		return { query, semantic: [], bm25: [], hybridScored: [] };
	}

	const pipeline = getHybridPipeline();
	const sharedOptions = {
		...options,
		query,
		topK: topK * pipeline.retrievalMultiplier
	};
	const [semanticSearch, bm25Search] = await Promise.all([
		searchSemantic(sharedOptions),
		searchBm25(sharedOptions)
	]);
	const candidates =
		pipeline.fusion === 'rrf'
			? rrfCandidates(semanticSearch.results, bm25Search.results, topK, pipeline)
			: unionCandidates(semanticSearch.results, bm25Search.results);
	const ranked = pipeline.reranker
		? await rerankMatches(query, candidates, pipeline.reranker, pipeline)
		: candidates;

	return {
		query,
		semantic: semanticSearch.results.map(withoutScore).slice(0, topK),
		bm25: bm25Search.results.map(withoutScore).slice(0, topK),
		hybridScored: orderByDocument(ranked, pipeline.documentOrder, topK)
	};
}

export async function searchAllMethods(options: SearchOptionsBase): Promise<SearchMethodResults> {
	const search = await collectMethodResults(options);
	const { settings } = await getActiveEmbedding();

	return {
		query: search.query,
		semantic: search.semantic,
		bm25: search.bm25,
		hybrid: search.hybridScored.map(withoutScore),
		pipeline: getHybridPipeline(),
		embeddingModel: settings.model,
		pdfPageCleanup
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
