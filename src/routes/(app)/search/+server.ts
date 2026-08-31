import { json } from '@sveltejs/kit';
import { DEFAULT_ASSISTANT_CONFIG } from '$lib/constants';
import { RetrievalMode } from '$lib/enums';
import { diagnosticEvents } from '$lib/server/diagnostics/events';
import { EMBEDDING_MODEL } from '$lib/server/rag/embedding-model';
import { searchAllMethodsWithTrace } from '$lib/server/rag/search/hybrid-search';
import {
	CROSS_ENCODER_MODEL,
	RETRIEVAL_SCORING_VERSION
} from '$lib/server/rag/search/retrieval-version';
import {
	RetrievalFeedbackRepository,
	RetrievalImpressionsRepository
} from '$lib/server/repositories';
import type { ApiSearchMatch, ApiSearchResults } from '$lib/types';
import type { RequestHandler } from './$types';

type UnratedSearchMatch = Omit<ApiSearchMatch, 'rating'>;

function attachRatings(
	matches: UnratedSearchMatch[],
	ratings: ReadonlyMap<string, ApiSearchMatch['rating']>
): ApiSearchMatch[] {
	return matches.map((match) => ({
		...match,
		rating: ratings.get(match.chunkId) ?? null
	}));
}

export const GET: RequestHandler = async ({ url }) => {
	const query = url.searchParams.get('query') ?? '';
	const requestedTopK = Number.parseInt(
		url.searchParams.get('topK') ?? String(DEFAULT_ASSISTANT_CONFIG.ragTopK),
		10
	);
	const topK = Number.isFinite(requestedTopK)
		? Math.min(100, Math.max(1, requestedTopK))
		: DEFAULT_ASSISTANT_CONFIG.ragTopK;
	const documentIds = url.searchParams.getAll('documentIds');
	const docs = documentIds.length > 0 ? documentIds : undefined;

	if (!query.trim()) {
		return json({
			bm25: [],
			semantic: [],
			hybrid: []
		} satisfies ApiSearchResults);
	}

	const started = Date.now();

	try {
		const execution = await searchAllMethodsWithTrace({
			query,
			topK,
			documentIds: docs
		});

		await RetrievalImpressionsRepository.record({
			query: execution.results.query,
			requestedTopK: topK,
			documentIds,
			embeddingModel: EMBEDDING_MODEL,
			rerankerModel: CROSS_ENCODER_MODEL,
			scoringVersion: RETRIEVAL_SCORING_VERSION,
			candidates: execution.candidates
		});

		const data = execution.results;
		const chunkIds = [...data.semantic, ...data.bm25, ...data.hybrid].map(({ chunkId }) => chunkId);
		const ratings = await RetrievalFeedbackRepository.findRatings(query, chunkIds);

		const response: ApiSearchResults = {
			[RetrievalMode.SEMANTIC]: attachRatings(data.semantic, ratings),
			[RetrievalMode.BM25]: attachRatings(data.bm25, ratings),
			[RetrievalMode.HYBRID]: attachRatings(data.hybrid, ratings)
		};

		diagnosticEvents.searchCompleted({
			durationMs: Date.now() - started,
			resultCount: data.hybrid.length,
			searchMode: 'all'
		});

		return json(response);
	} catch (error) {
		console.error('[Search] Search Context retrieval failed.', error);
		diagnosticEvents.searchFailed('all');

		return json(
			{
				error: 'Search failed.'
			},
			{ status: 500 }
		);
	}
};
