import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { DEFAULT_RELEX_THRESHOLDS } from '../../src/lib/constants/entity-extraction';
import { extractRelex, loadRelexRuntime } from '../../src/lib/server/entities/relex';
import { evidence } from './evidence.mts';

interface Labels {
	entityLabels: string[];
	relationLabels: string[];
}

interface Chunk {
	id: string;
	source: string;
	text: string;
}

const { values } = parseArgs({
	options: {
		labels: { type: 'string', default: 'benchmarks/relex-eval/labels.json' },
		chunks: { type: 'string', default: '.cache/relex-eval/own-chunks.json' },
		out: { type: 'string', default: '.cache/relex-eval/own-predictions.json' },
		'relation-threshold': { type: 'string' },
		'per-document': { type: 'string' }
	}
});

const thresholds = {
	...DEFAULT_RELEX_THRESHOLDS,
	...(values['relation-threshold'] ? { relation: Number(values['relation-threshold']) } : {})
};
const labels = JSON.parse(readFileSync(values.labels, 'utf8')) as Labels;
const perDocument = values['per-document']
	? (JSON.parse(readFileSync(values['per-document'], 'utf8')) as Record<string, Labels>)
	: undefined;
const chunks = JSON.parse(readFileSync(values.chunks, 'utf8')) as Chunk[];
await loadRelexRuntime();

const results = [];
const skipped: { id: string; source: string; reason: string }[] = [];
for (const [index, chunk] of chunks.entries()) {
	const chunkLabels = perDocument ? perDocument[chunk.source] : labels;
	if (!chunkLabels?.entityLabels.length || !chunkLabels.relationLabels.length) {
		const reason = chunkLabels ? 'no entity or relation labels' : 'document not in the label map';
		skipped.push({ id: chunk.id, source: chunk.source, reason });
		console.log(`${index + 1}/${chunks.length} ${chunk.source}: skipped, ${reason}`);
		continue;
	}
	const started = performance.now();
	const result = await extractRelex(chunk.text, { ...chunkLabels, thresholds });
	results.push({
		id: chunk.id,
		source: chunk.source,
		labels: chunkLabels,
		seconds: (performance.now() - started) / 1000,
		entities: result.entities.map(({ text, label, score }) => ({ text, label, score })),
		relations: result.relations.map((relation) => ({
			head: relation.head.text,
			headLabel: relation.head.label,
			relation: relation.relation,
			tail: relation.tail.text,
			tailLabel: relation.tail.label,
			score: relation.score,
			evidence: evidence(chunk.text, relation.head, relation.tail)
		}))
	});
	console.log(
		`${index + 1}/${chunks.length} ${chunk.source}: ${result.entities.length} entities, ${result.relations.length} relations`
	);
}

writeFileSync(
	values.out,
	JSON.stringify({ labels: perDocument ?? labels, thresholds, skipped, chunks: results }, null, 2) +
		'\n'
);
console.log(`Done: ${values.out}`);
