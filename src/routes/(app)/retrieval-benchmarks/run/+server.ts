import { json } from '@sveltejs/kit';
import { runFixedRetrievalBenchmark } from '$lib/server/rag/benchmark';
import type { ApiRetrievalBenchmarkReport } from '$lib/types';
import type { RequestHandler } from './$types';

const DEFAULT_BENCHMARK_TOP_K = 20;

export const GET: RequestHandler = async ({ url }) => {
	const requestedTopK = Number.parseInt(
		url.searchParams.get('topK') ?? String(DEFAULT_BENCHMARK_TOP_K),
		10
	);
	const topK = Number.isFinite(requestedTopK)
		? Math.min(100, Math.max(1, requestedTopK))
		: DEFAULT_BENCHMARK_TOP_K;

	try {
		const report = await runFixedRetrievalBenchmark(topK);
		return json(report satisfies ApiRetrievalBenchmarkReport);
	} catch (error) {
		console.error('[Retrieval Benchmark] Benchmark run failed.', error);

		return json(
			{
				error: error instanceof Error ? error.message : 'Retrieval benchmark failed.'
			},
			{ status: 400 }
		);
	}
};
