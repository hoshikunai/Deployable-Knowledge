import { resolve } from 'node:path';
import { databaseClient } from '../src/lib/server/database/database';
import { loadKnowledgeGraph } from '../src/lib/server/knowledge-graph-new/knowledge-graph';
import {
	searchKnowledgeGraph,
	type GraphSearchResult
} from '../src/lib/server/knowledge-graph-new/search';
import { evaluateAssertions } from '../tests/knowledge-graph/assertion-evaluator';
import {
	loadGoldBenchmark,
	validateCorpusChunks,
	validateGoldBenchmark,
	type CorpusChunk
} from '../tests/knowledge-graph/benchmark-validation';
import { evaluateRetrieval } from '../tests/knowledge-graph/retrieval-evaluator';

const fixtureArgument = process.argv.slice(2).find((argument) => !argument.startsWith('--'));
const fixturePath = resolve(fixtureArgument ?? 'tests/knowledge-graph/fixtures/benchmark-v1.json');
const assertionsOnly = process.argv.includes('--assertions-only');
const benchmark = await loadGoldBenchmark(fixturePath);
const fixtureErrors = validateGoldBenchmark(benchmark);
if (fixtureErrors.length) fail('The gold benchmark is invalid:', fixtureErrors);

const corpusChunks = await loadCorpusChunks(benchmark.chunks.map((chunk) => chunk.chunkId));
const corpusErrors = validateCorpusChunks(benchmark, corpusChunks);
if (corpusErrors.length) fail('The benchmark no longer matches the stored corpus:', corpusErrors);

const graph = await loadKnowledgeGraph();
const assertions = evaluateAssertions(benchmark, graph.assertions);
const report: Record<string, unknown> = {
	benchmark: {
		version: benchmark.version,
		corpusId: benchmark.corpusId,
		chunks: benchmark.chunks.length,
		goldAssertions: benchmark.assertions.length,
		queries: benchmark.queries.length
	},
	assertions
};

if (!assertionsOnly) {
	const maxK = 10;
	const results = new Map<string, GraphSearchResult>();
	for (const query of benchmark.queries) {
		results.set(
			query.id,
			await searchKnowledgeGraph({
				query: query.question,
				topK: maxK,
				usePpr: true
			})
		);
	}
	report.retrieval = evaluateRetrieval(benchmark, results, assertions.matches);
}

console.log(JSON.stringify(report, null, 2));
await databaseClient.close();

async function loadCorpusChunks(chunkIds: string[]): Promise<CorpusChunk[]> {
	if (!chunkIds.length) return [];
	const placeholders = chunkIds.map(() => '?').join(', ');
	const rows = await databaseClient.execute({
		sql: `SELECT id, document_id, content FROM document_chunks WHERE id IN (${placeholders})`,
		args: chunkIds
	});
	return rows.rows.map((row) => ({
		chunkId: String(row.id),
		documentId: String(row.document_id),
		content: String(row.content)
	}));
}

function fail(heading: string, errors: string[]): never {
	throw new Error(`${heading}\n- ${errors.join('\n- ')}`);
}
