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

type SearchMatchWithoutFeedback = Omit<ApiSearchMatch, 'impressionResultId' | 'rating'>;

function resultKey(retrievalMode: RetrievalMode, chunkId: string): string {
	return `${retrievalMode}\u0000${chunkId}`;
}

function attachFeedback(
	retrievalMode: RetrievalMode,
	matches: SearchMatchWithoutFeedback[],
	ratings: ReadonlyMap<string, ApiSearchMatch['rating']>,
	impressionResultIds: ReadonlyMap<string, string>
): ApiSearchMatch[] {
	return matches.map((match) => {
		const impressionResultId = impressionResultIds.get(resultKey(retrievalMode, match.chunkId));

		if (!impressionResultId) {
			throw new Error(`Missing impression result for ${retrievalMode}:${match.chunkId}.`);
		}

		return {
			...match,
			impressionResultId,
			rating: ratings.get(match.chunkId) ?? null
		};
	});
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

		const recordedImpression = await RetrievalImpressionsRepository.record({
			query: execution.results.query,
			requestedTopK: topK,
			documentIds,
			embeddingModel: EMBEDDING_MODEL,
			rerankerModel: CROSS_ENCODER_MODEL,
			scoringVersion: RETRIEVAL_SCORING_VERSION,
			candidates: execution.candidates
		});

		const impressionResultIds = new Map(
			recordedImpression.results.map((result) => [
				resultKey(result.retrievalMode, result.chunkId),
				result.id
			])
		);

		const data = execution.results;
		const chunkIds = [...data.semantic, ...data.bm25, ...data.hybrid].map(({ chunkId }) => chunkId);
		const ratings = await RetrievalFeedbackRepository.findRatings(query, chunkIds);

		const response: ApiSearchResults = {
			[RetrievalMode.SEMANTIC]: attachFeedback(
				RetrievalMode.SEMANTIC,
				data.semantic,
				ratings,
				impressionResultIds
			),
			[RetrievalMode.BM25]: attachFeedback(
				RetrievalMode.BM25,
				data.bm25,
				ratings,
				impressionResultIds
			),
			[RetrievalMode.HYBRID]: attachFeedback(
				RetrievalMode.HYBRID,
				data.hybrid,
				ratings,
				impressionResultIds
			)
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
