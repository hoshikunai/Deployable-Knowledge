// Hybrid search gathers semantic and BM25 candidates, then reranks them together.

import { RetrievalMode } from '$lib/enums';
import { searchSemantic } from './semantic-search';
import { searchBm25 } from './bm25-search';
import { rerankCandidates } from './cross-rerank';
import { feedbackCandidateLimit, rerankWithRetrievalFeedback } from './feedback-rerank';
import {
	buildRetrievalCandidateSnapshots,
	type RetrievalCandidateSnapshot,
	type ScoredSearchMatch,
	type SearchMatchBase,
	type SearchOptionsBase,
	type SearchResult
} from './search-shared';

export interface SearchMethodResults {
	query: string;
	semantic: SearchMatchBase[];
	bm25: SearchMatchBase[];
	hybrid: SearchMatchBase[];
}

export interface SearchMethodExecution {
	results: SearchMethodResults;
	candidates: RetrievalCandidateSnapshot[];
}

interface CollectedMethodResults {
	query: string;
	semanticScored: ScoredSearchMatch[];
	bm25Scored: ScoredSearchMatch[];
	hybridScored: ScoredSearchMatch[];
	crossEncoderScores: Map<string, number>;
}

function withoutScore(match: ScoredSearchMatch): SearchMatchBase {
	const { score: _score, ...chunk } = match;
	return chunk;
}

async function collectMethodResults(options: SearchOptionsBase): Promise<CollectedMethodResults> {
	const query = options.query.trim();
	const topK = Math.max(0, Math.floor(options.topK ?? 10));

	if (!query || topK === 0) {
		return {
			query,
			semanticScored: [],
			bm25Scored: [],
			hybridScored: [],
			crossEncoderScores: new Map()
		};
	}

	const sharedOptions = {
		...options,
		query,
		topK: topK * 2
	};

	const [semanticSearch, bm25Search] = await Promise.all([
		searchSemantic(sharedOptions),
		searchBm25(sharedOptions)
	]);

	const semanticScored = semanticSearch.results.slice(0, topK);
	const bm25Scored = bm25Search.results.slice(0, topK);
	const byChunkId = new Map<string, SearchMatchBase>();

	for (const match of [...semanticScored, ...bm25Scored]) {
		if (!byChunkId.has(match.chunkId)) {
			byChunkId.set(match.chunkId, withoutScore(match));
		}
	}

	const rankedCandidates = await rerankCandidates(
		query,
		[...byChunkId.values()].map((match) => ({
			chunkId: match.chunkId,
			content: match.content
		}))
	);

	const crossEncoderScores = new Map(
		rankedCandidates.map((candidate) => [candidate.chunkId, candidate.relevance])
	);

	const hybridScored: ScoredSearchMatch[] = [];

	for (const candidate of rankedCandidates) {
		const match = byChunkId.get(candidate.chunkId);
		if (match) {
			hybridScored.push({
				...match,
				score: candidate.relevance
			});
		}

		if (hybridScored.length === topK) break;
	}

	return {
		query,
		semanticScored,
		bm25Scored,
		hybridScored,
		crossEncoderScores
	};
}

export async function searchAllMethodsWithTrace(
	options: SearchOptionsBase
): Promise<SearchMethodExecution> {
	const topK = Math.max(0, Math.floor(options.topK ?? 10));
	const search = await collectMethodResults({
		...options,
		topK: feedbackCandidateLimit(topK)
	});

	const [semantic, bm25, hybrid] = await Promise.all([
		rerankWithRetrievalFeedback(search.query, search.semanticScored, topK),
		rerankWithRetrievalFeedback(search.query, search.bm25Scored, topK),
		rerankWithRetrievalFeedback(search.query, search.hybridScored, topK)
	]);

	const scoreMaps = {
		semantic: new Map(search.semanticScored.map((match) => [match.chunkId, match.score])),
		bm25: new Map(search.bm25Scored.map((match) => [match.chunkId, match.score])),
		crossEncoder: search.crossEncoderScores
	};

	return {
		results: {
			query: search.query,
			semantic: semantic.map(withoutScore),
			bm25: bm25.map(withoutScore),
			hybrid: hybrid.map(withoutScore)
		},
		candidates: [
			...buildRetrievalCandidateSnapshots(
				RetrievalMode.SEMANTIC,
				search.semanticScored,
				semantic,
				scoreMaps
			),
			...buildRetrievalCandidateSnapshots(RetrievalMode.BM25, search.bm25Scored, bm25, scoreMaps),
			...buildRetrievalCandidateSnapshots(
				RetrievalMode.HYBRID,
				search.hybridScored,
				hybrid,
				scoreMaps
			)
		]
	};
}

export async function searchAllMethods(options: SearchOptionsBase): Promise<SearchMethodResults> {
	const execution = await searchAllMethodsWithTrace(options);
	return execution.results;
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
