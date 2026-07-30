import { error, json } from '@sveltejs/kit';
import { DEFAULT_ASSISTANT_CONFIG } from '$lib/constants';
import {
	isRetrievalMode,
	searchWithRetrievalMode
} from '$lib/server/rag/search/retrieval-registry';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
	const query = url.searchParams.get('query') ?? '';
	const requestedTopK = Number.parseInt(
		url.searchParams.get('topK') ?? String(DEFAULT_ASSISTANT_CONFIG.ragTopK),
		10
	);
	const topK = Number.isFinite(requestedTopK)
		? Math.max(1, requestedTopK)
		: DEFAULT_ASSISTANT_CONFIG.ragTopK;
	const documentIds = url.searchParams.getAll('documentIds');
	const docs = documentIds.length ? documentIds : undefined;
	const requestedMode = url.searchParams.get('mode') ?? '';
	if (requestedMode && !isRetrievalMode(requestedMode)) {
		throw error(400, `Unsupported retrieval mode: ${requestedMode}`);
	}
	const mode = isRetrievalMode(requestedMode)
		? requestedMode
		: DEFAULT_ASSISTANT_CONFIG.retrievalMode;

	if (!query.trim()) {
		return json([]);
	}

	const matches = await searchWithRetrievalMode(mode, {
		query,
		topK,
		documentIds: docs
	});
	return json(
		matches.map(
			({ chunkId, documentId, sourceTitle, sourceType, pageIndex, chunkIndex, content }) => ({
				chunkId,
				documentId,
				sourceTitle,
				sourceType,
				pageIndex,
				chunkIndex,
				content
			})
		)
	);
};
