import type { GraphSearchResult } from '../../src/lib/server/knowledge-graph-new/search';
import type { GoldBenchmark, GoldQuery } from './benchmark-types';
import type { AssertionMatch } from './assertion-evaluator';

export interface QueryEvaluation {
	queryId: string;
	reciprocalRank: number;
	pathRecall: number | null;
	metricsAtK: Record<
		string,
		{
			precision: number;
			recall: number;
			hit: number;
			ndcg: number;
			forbiddenHit: number;
		}
	>;
}

export interface RetrievalEvaluation {
	queryCount: number;
	meanReciprocalRank: number;
	meanPathRecall: number;
	metricsAtK: QueryEvaluation['metricsAtK'];
	queries: QueryEvaluation[];
}

export function evaluateRetrieval(
	benchmark: GoldBenchmark,
	results: Map<string, GraphSearchResult>,
	assertionMatches: AssertionMatch[],
	cutoffs: number[] = [1, 3, 5, 10]
): RetrievalEvaluation {
	const normalizedCutoffs = [...new Set(cutoffs)]
		.filter((cutoff) => cutoff > 0)
		.sort((a, b) => a - b);
	if (!normalizedCutoffs.length)
		throw new Error('At least one positive retrieval cutoff is required.');
	const actualToGold = new Map(
		assertionMatches.map((match) => [match.actualAssertionId, match.goldAssertionId])
	);
	const queries = benchmark.queries.map((query) =>
		evaluateQuery(query, requiredResult(results, query.id), actualToGold, normalizedCutoffs)
	);

	return {
		queryCount: queries.length,
		meanReciprocalRank: average(queries.map((query) => query.reciprocalRank)),
		meanPathRecall: average(
			queries.flatMap((query) => (query.pathRecall === null ? [] : [query.pathRecall]))
		),
		metricsAtK: Object.fromEntries(
			normalizedCutoffs.map((cutoff) => {
				const key = String(cutoff);
				return [
					key,
					{
						precision: average(queries.map((query) => query.metricsAtK[key].precision)),
						recall: average(queries.map((query) => query.metricsAtK[key].recall)),
						hit: average(queries.map((query) => query.metricsAtK[key].hit)),
						ndcg: average(queries.map((query) => query.metricsAtK[key].ndcg)),
						forbiddenHit: average(queries.map((query) => query.metricsAtK[key].forbiddenHit))
					}
				];
			})
		),
		queries
	};
}

function evaluateQuery(
	query: GoldQuery,
	result: GraphSearchResult,
	actualToGold: Map<string, string>,
	cutoffs: number[]
): QueryEvaluation {
	const rankedChunkIds = unique(result.chunks.map((chunk) => chunk.chunkId));
	const relevant = new Set(query.relevantChunkIds);
	const forbidden = new Set(query.forbiddenChunkIds ?? []);
	const firstRelevantIndex = rankedChunkIds.findIndex((chunkId) => relevant.has(chunkId));
	const returnedGoldPaths = result.paths.map((path) =>
		path.assertionIds.map(
			(assertionId) => actualToGold.get(assertionId) ?? `unmatched:${assertionId}`
		)
	);
	const pathHits = query.expectedPathAssertionIds.filter((expected) =>
		returnedGoldPaths.some((actual) => samePath(actual, expected))
	).length;

	return {
		queryId: query.id,
		reciprocalRank: firstRelevantIndex < 0 ? 0 : 1 / (firstRelevantIndex + 1),
		pathRecall:
			query.expectedPathAssertionIds.length === 0
				? null
				: pathHits / query.expectedPathAssertionIds.length,
		metricsAtK: Object.fromEntries(
			cutoffs.map((cutoff) => {
				const retrieved = rankedChunkIds.slice(0, cutoff);
				const relevantCount = retrieved.filter((chunkId) => relevant.has(chunkId)).length;
				return [
					String(cutoff),
					{
						precision: relevantCount / cutoff,
						recall: ratio(relevantCount, relevant.size),
						hit: relevantCount > 0 ? 1 : 0,
						ndcg: ndcg(retrieved, relevant),
						forbiddenHit: retrieved.some((chunkId) => forbidden.has(chunkId)) ? 1 : 0
					}
				];
			})
		)
	};
}

function requiredResult(
	results: Map<string, GraphSearchResult>,
	queryId: string
): GraphSearchResult {
	const result = results.get(queryId);
	if (!result) throw new Error(`Missing search result for query ${queryId}.`);
	return result;
}

function ndcg(retrieved: string[], relevant: Set<string>): number {
	const dcg = retrieved.reduce(
		(total, chunkId, index) => total + (relevant.has(chunkId) ? 1 / Math.log2(index + 2) : 0),
		0
	);
	const idealCount = Math.min(retrieved.length, relevant.size);
	const ideal = Array.from({ length: idealCount }, (_, index) => 1 / Math.log2(index + 2)).reduce(
		(total, score) => total + score,
		0
	);
	return ratio(dcg, ideal);
}

function samePath(actual: string[], expected: string[]): boolean {
	return actual.length === expected.length && actual.every((id, index) => id === expected[index]);
}

function unique(values: string[]): string[] {
	return [...new Set(values)];
}

function average(values: number[]): number {
	return ratio(
		values.reduce((total, value) => total + value, 0),
		values.length
	);
}

function ratio(numerator: number, denominator: number): number {
	return denominator === 0 ? 0 : numerator / denominator;
}
