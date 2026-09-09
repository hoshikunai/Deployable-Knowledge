import { searchForRetrievalBenchmark } from '$lib/server/rag/search/hybrid-search';
import type { ScoredSearchMatch } from '$lib/server/rag/search/search-shared';
import { RetrievalBenchmarksRepository } from '$lib/server/repositories';
import {
	RETRIEVAL_BENCHMARK_RANKINGS,
	type RetrievalBenchmarkCaseResult,
	type RetrievalBenchmarkJudgment,
	type RetrievalBenchmarkMetrics,
	type RetrievalBenchmarkRanking,
	type RetrievalBenchmarkReport
} from './retrieval-benchmark.types';

const RELEVANT_RATING_THRESHOLD = 4;
const NDCG_LIMIT = 5;

function relevanceGain(relevance: number): number {
	// Unjudged chunks use 0 and contribute no relevance gain.
	if (relevance === 0) return 0;

	return 2 ** (relevance - 1) - 1;
}

function discountedCumulativeGain(relevances: number[]): number {
	return relevances.slice(0, NDCG_LIMIT).reduce((total, relevance, index) => {
		return total + relevanceGain(relevance) / Math.log2(index + 2);
	}, 0);
}

function calculateMetrics(
	results: ScoredSearchMatch[],
	judgments: RetrievalBenchmarkJudgment[],
	topK: number
): RetrievalBenchmarkMetrics {
	const relevanceByChunkId = new Map(
		judgments.map((judgment) => [judgment.chunkId, judgment.relevance])
	);
	const relevantChunkIds = new Set(
		judgments
			.filter((judgment) => judgment.relevance >= RELEVANT_RATING_THRESHOLD)
			.map((judgment) => judgment.chunkId)
	);
	const limitedResults = results.slice(0, topK);
	const retrievedRelevantChunkIds = new Set(
		limitedResults
			.filter((result) => relevantChunkIds.has(result.chunkId))
			.map((result) => result.chunkId)
	);

	let recallAtK = 0;
	if (relevantChunkIds.size > 0) {
		recallAtK = retrievedRelevantChunkIds.size / relevantChunkIds.size;
	}

	const firstRelevantIndex = limitedResults.findIndex((result) =>
		relevantChunkIds.has(result.chunkId)
	);
	const reciprocalRank = firstRelevantIndex === -1 ? 0 : 1 / (firstRelevantIndex + 1);

	const retrievedRelevances = limitedResults.map(
		(result) => relevanceByChunkId.get(result.chunkId) ?? 0
	);
	const idealRelevances = judgments
		.map((judgment) => judgment.relevance)
		.sort((left, right) => right - left);

	const idealDcg = discountedCumulativeGain(idealRelevances);
	const ndcgAt5 = idealDcg === 0 ? 0 : discountedCumulativeGain(retrievedRelevances) / idealDcg;

	return {
		recallAtK,
		reciprocalRank,
		ndcgAt5
	};
}

function averageMetrics(
	cases: RetrievalBenchmarkCaseResult[],
	ranking: RetrievalBenchmarkRanking
): RetrievalBenchmarkMetrics {
	if (cases.length === 0) {
		return {
			recallAtK: 0,
			reciprocalRank: 0,
			ndcgAt5: 0
		};
	}

	const totals = cases.reduce(
		(current, benchmarkCase) => ({
			recallAtK: current.recallAtK + benchmarkCase.metrics[ranking].recallAtK,
			reciprocalRank: current.reciprocalRank + benchmarkCase.metrics[ranking].reciprocalRank,
			ndcgAt5: current.ndcgAt5 + benchmarkCase.metrics[ranking].ndcgAt5
		}),
		{
			recallAtK: 0,
			reciprocalRank: 0,
			ndcgAt5: 0
		}
	);

	return {
		recallAtK: totals.recallAtK / cases.length,
		reciprocalRank: totals.reciprocalRank / cases.length,
		ndcgAt5: totals.ndcgAt5 / cases.length
	};
}

export async function runFixedRetrievalBenchmark(topK: number): Promise<RetrievalBenchmarkReport> {
	const benchmarkCases = await RetrievalBenchmarksRepository.list();

	if (benchmarkCases.length === 0) {
		throw new Error('The fixed retrieval benchmark does not contain any cases.');
	}

	const cases: RetrievalBenchmarkCaseResult[] = [];
	let activeModelId: string | null = null;

	for (const benchmarkCase of benchmarkCases) {
		const execution = await searchForRetrievalBenchmark({
			query: benchmarkCase.query,
			topK,
			documentIds: benchmarkCase.documentIds.length > 0 ? benchmarkCase.documentIds : undefined
		});

		if (cases.length > 0 && activeModelId !== execution.activeModelId) {
			throw new Error('The learned model changed during the benchmark. Run the benchmark again.');
		}
		activeModelId = execution.activeModelId;

		cases.push({
			caseId: benchmarkCase.id,
			name: benchmarkCase.name,
			query: benchmarkCase.query,
			metrics: {
				semantic: calculateMetrics(execution.rankings.semantic, benchmarkCase.judgments, topK),
				bm25: calculateMetrics(execution.rankings.bm25, benchmarkCase.judgments, topK),
				hybridBaseline: calculateMetrics(
					execution.rankings.hybridBaseline,
					benchmarkCase.judgments,
					topK
				),
				hybridLearned: calculateMetrics(
					execution.rankings.hybridLearned,
					benchmarkCase.judgments,
					topK
				)
			}
		});
	}

	return {
		generatedAt: new Date().toISOString(),
		topK,
		caseCount: cases.length,
		activeModelId,
		cases,
		aggregate: {
			semantic: averageMetrics(cases, RETRIEVAL_BENCHMARK_RANKINGS[0]),
			bm25: averageMetrics(cases, RETRIEVAL_BENCHMARK_RANKINGS[1]),
			hybridBaseline: averageMetrics(cases, RETRIEVAL_BENCHMARK_RANKINGS[2]),
			hybridLearned: averageMetrics(cases, RETRIEVAL_BENCHMARK_RANKINGS[3])
		}
	};
}
