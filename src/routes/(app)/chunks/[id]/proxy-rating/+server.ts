import { json, type RequestHandler } from '@sveltejs/kit';
import { AI_PROXY_FEEDBACK_SOURCE, CHUNK_RATING_VALUES } from '$lib/constants';
import { RetrievalFeedbackRepository } from '$lib/server/repositories';
import type {
	ApiAiProxyChunkRatingRequest,
	ApiAiProxyChunkRatingResponse,
	ApiChunkRatingDeleteRequest,
	ChunkRatingValue
} from '$lib/types';

const MAXIMUM_QUERY_LENGTH = 2_000;
const MAXIMUM_RATIONALE_LENGTH = 1_000;

function isRequestObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readRequestObject(request: Request): Promise<Record<string, unknown> | null> {
	try {
		const value: unknown = await request.json();
		return isRequestObject(value) ? value : null;
	} catch {
		return null;
	}
}

function isChunkRating(value: unknown): value is ChunkRatingValue {
	return typeof value === 'number' && CHUNK_RATING_VALUES.includes(value as ChunkRatingValue);
}

export const PATCH: RequestHandler = async ({ params, request }) => {
	const chunkId = params.id;
	if (!chunkId) return json({ error: 'Missing chunk id.' }, { status: 400 });

	const body = await readRequestObject(request);
	if (!body) return json({ error: 'A JSON request body is required.' }, { status: 400 });

	const query = typeof body.query === 'string' ? body.query.trim() : '';
	const impressionResultId =
		typeof body.impressionResultId === 'string' ? body.impressionResultId.trim() : '';
	const rating = body.rating;
	const confidence = body.confidence;
	const rationale = typeof body.rationale === 'string' ? body.rationale.trim() : '';

	if (!query || query.length > MAXIMUM_QUERY_LENGTH) {
		return json({ error: 'A valid search query is required.' }, { status: 400 });
	}

	if (!isChunkRating(rating)) {
		return json({ error: 'Rating must be an integer from 1 through 5.' }, { status: 400 });
	}

	if (!impressionResultId || impressionResultId.length > 100) {
		return json({ error: 'A valid impression result id is required.' }, { status: 400 });
	}

	if (
		typeof confidence !== 'number' ||
		!Number.isFinite(confidence) ||
		confidence < 0 ||
		confidence > 1
	) {
		return json({ error: 'Confidence must be a number from zero through one.' }, { status: 400 });
	}

	if (!rationale || rationale.length > MAXIMUM_RATIONALE_LENGTH) {
		return json({ error: 'A concise rating rationale is required.' }, { status: 400 });
	}

	const input: ApiAiProxyChunkRatingRequest = {
		impressionResultId,
		query,
		rating,
		confidence,
		rationale
	};
	const row = await RetrievalFeedbackRepository.setAiProxy({
		chunkId,
		...input
	});

	if (!row) return json({ error: 'The referenced search result was not found.' }, { status: 404 });

	return json({
		chunkId,
		rating: row.rating as ChunkRatingValue,
		feedbackSource: AI_PROXY_FEEDBACK_SOURCE,
		confidence: row.confidence,
		rationale: row.rationale
	} satisfies ApiAiProxyChunkRatingResponse);
};

export const DELETE: RequestHandler = async ({ params, request }) => {
	const chunkId = params.id;
	if (!chunkId) return json({ error: 'Missing chunk id.' }, { status: 400 });

	const body = await readRequestObject(request);
	if (!body) return json({ error: 'A JSON request body is required.' }, { status: 400 });

	const query = typeof body.query === 'string' ? body.query.trim() : '';
	if (!query || query.length > MAXIMUM_QUERY_LENGTH) {
		return json({ error: 'A valid search query is required.' }, { status: 400 });
	}

	const input: ApiChunkRatingDeleteRequest = { query };
	await RetrievalFeedbackRepository.clearAiProxy(chunkId, input.query);

	return json({
		chunkId,
		rating: null,
		feedbackSource: AI_PROXY_FEEDBACK_SOURCE,
		confidence: null,
		rationale: null
	} satisfies ApiAiProxyChunkRatingResponse);
};
