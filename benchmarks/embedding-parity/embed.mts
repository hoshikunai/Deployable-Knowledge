// Embeds texts.json with the app's own EmbeddingModel (selected by RAG_EMBEDDING_MODEL) and
// writes the vectors for compare.py. Run from the repository root so the model cache resolves:
//   RAG_EMBEDDING_MODEL=bge-base-en-v1.5 node_modules/.bin/tsx benchmarks/embedding-parity/embed.mts out.json

import { readFileSync, writeFileSync } from 'node:fs';
import { activeEmbeddingModel } from '../../src/lib/server/rag/embedding-models/registry';

const [output] = process.argv.slice(2);
if (!output) throw new Error('Usage: embed.mts <output.json>');

const texts = JSON.parse(readFileSync(new URL('./texts.json', import.meta.url), 'utf8')) as {
	queries: string[];
	passages: string[];
};

const started = Date.now();
const queries = await activeEmbeddingModel.embed(texts.queries, 'search_query');
const passages = await activeEmbeddingModel.embed(texts.passages, 'search_document');

writeFileSync(
	output,
	JSON.stringify({
		model: activeEmbeddingModel.id,
		seconds: (Date.now() - started) / 1000,
		queries: queries.map((vector) => Array.from(vector)),
		passages: passages.map((vector) => Array.from(vector))
	})
);
console.log(`${activeEmbeddingModel.id}: ${queries[0].length} dimensions -> ${output}`);
