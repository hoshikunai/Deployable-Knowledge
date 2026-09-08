import type { ChunkRatingValue } from '$lib/types';

export const RETRIEVAL_BENCHMARK_RANKINGS = [
	'semantic',
	'bm25',
	'hybridBaseline',
	'hybridLearned'
] as const;

export type RetrievalBenchmarkRanking = (typeof RETRIEVAL_BENCHMARK_RANKINGS)[number];

export interface RetrievalBenchmarkJudgment {
	chunkId: string;
	relevance: ChunkRatingValue;
}

export interface RetrievalBenchmarkCase {
	id: string;
	name: string;
	query: string;
	documentIds: string[];
	judgments: RetrievalBenchmarkJudgment[];
	createdAt: string;
}

export interface CreateRetrievalBenchmarkCaseInput {
	name: string;
	query: string;
	documentIds: string[];
	judgments: RetrievalBenchmarkJudgment[];
}

export interface RetrievalBenchmarkMetrics {
	recallAtK: number;
	reciprocalRank: number;
	ndcgAt5: number;
}

export interface RetrievalBenchmarkCaseResult {
	caseId: string;
	name: string;
	query: string;
	metrics: Record<RetrievalBenchmarkRanking, RetrievalBenchmarkMetrics>;
}

export interface RetrievalBenchmarkReport {
	generatedAt: string;
	topK: number;
	caseCount: number;
	activeModelId: string | null;
	cases: RetrievalBenchmarkCaseResult[];
	aggregate: Record<RetrievalBenchmarkRanking, RetrievalBenchmarkMetrics>;
}
