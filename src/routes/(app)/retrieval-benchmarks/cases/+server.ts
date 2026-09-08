import { json } from '@sveltejs/kit';
import { CHUNK_RATING_VALUES } from '$lib/constants';
import { RetrievalBenchmarksRepository } from '$lib/server/repositories';
import type {
	ApiRetrievalBenchmarkCaseCreateRequest,
	ApiRetrievalBenchmarkCasesResponse,
	ChunkRatingValue
} from '$lib/types';
import type { RequestHandler } from './$types';

function isRequestObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readCreateRequest(
	request: Request
): Promise<ApiRetrievalBenchmarkCaseCreateRequest | null> {
	let value: unknown;

	try {
		value = await request.json();
	} catch {
		return null;
	}

	if (!isRequestObject(value)) return null;
	if (typeof value.name !== 'string' || !value.name.trim()) return null;
	if (typeof value.query !== 'string' || !value.query.trim()) return null;
	if (!Array.isArray(value.judgments) || value.judgments.length === 0) return null;

	const rawDocumentIds = value.documentIds ?? [];
	if (
		!Array.isArray(rawDocumentIds) ||
		!rawDocumentIds.every((documentId) => typeof documentId === 'string')
	) {
		return null;
	}

	const judgments = [];
	const chunkIds = new Set<string>();

	for (const judgment of value.judgments) {
		if (!isRequestObject(judgment)) return null;
		if (typeof judgment.chunkId !== 'string' || !judgment.chunkId.trim()) return null;
		if (
			typeof judgment.relevance !== 'number' ||
			!CHUNK_RATING_VALUES.includes(judgment.relevance as ChunkRatingValue)
		) {
			return null;
		}

		const chunkId = judgment.chunkId.trim();
		if (chunkIds.has(chunkId)) return null;

		chunkIds.add(chunkId);
		judgments.push({
			chunkId,
			relevance: judgment.relevance as ChunkRatingValue
		});
	}

	if (!judgments.some((judgment) => judgment.relevance >= 4)) return null;

	return {
		name: value.name.trim(),
		query: value.query.trim(),
		documentIds: [
			...new Set(rawDocumentIds.map((documentId) => documentId.trim()).filter(Boolean))
		],
		judgments
	};
}

export const GET: RequestHandler = async () => {
	const cases = await RetrievalBenchmarksRepository.list();
	return json({ cases } satisfies ApiRetrievalBenchmarkCasesResponse);
};

export const POST: RequestHandler = async ({ request }) => {
	const input = await readCreateRequest(request);

	if (!input) {
		return json(
			{
				error:
					'Provide a name, query, optional documentIds, and unique 1-5 relevance judgments with at least one 4- or 5-star chunk.'
			},
			{ status: 400 }
		);
	}

	try {
		const benchmarkCase = await RetrievalBenchmarksRepository.create({
			name: input.name,
			query: input.query,
			documentIds: input.documentIds ?? [],
			judgments: input.judgments
		});

		return json(benchmarkCase, { status: 201 });
	} catch (error) {
		console.error('[Retrieval Benchmark] Could not create benchmark case.', error);
		return json(
			{ error: 'Could not create the benchmark case. Verify that every chunk ID exists.' },
			{ status: 400 }
		);
	}
};
