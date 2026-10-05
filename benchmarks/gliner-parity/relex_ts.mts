import { readFileSync, writeFileSync } from 'node:fs';
import {
	extractRelex,
	loadRelexRuntime,
	type RelexRelation
} from '../../src/lib/server/entities/relex';

const OUT = '.cache/gliner-parity';
const THRESHOLDS = { entity: 0.5, adjacency: 0.5, relation: 0.7 };
const EMPTY_TEXT = 'It rained for most of the afternoon and the streets stayed quiet.';

const texts = JSON.parse(readFileSync(`${OUT}/texts.json`, 'utf8')) as {
	entityLabels: string[];
	relations: string[];
	chunks: { id: string; text: string }[];
};

async function extract(text: string) {
	const result = await extractRelex(text, {
		entityLabels: texts.entityLabels,
		relationLabels: texts.relations,
		thresholds: THRESHOLDS
	});
	return {
		entities: result.entities,
		relations: result.relations.map((relation: RelexRelation) => ({
			head: [relation.head.start, relation.head.end],
			headText: relation.head.text,
			relation: relation.relation,
			tail: [relation.tail.start, relation.tail.end],
			tailText: relation.tail.text,
			score: relation.score
		}))
	};
}

async function attempt(name: string, action: () => Promise<unknown>) {
	try {
		return { case: name, ok: true, result: await action() };
	} catch (error) {
		return {
			case: name,
			ok: false,
			error: error instanceof Error ? `${error.name}: ${error.message}` : String(error)
		};
	}
}

const loadStarted = performance.now();
await loadRelexRuntime();
const loadSeconds = (performance.now() - loadStarted) / 1000;
await attempt('warm-up', () => extract(texts.chunks[0].text));

const chunks = [];
for (const [index, chunk] of texts.chunks.entries()) {
	const started = performance.now();
	const outcome = await attempt(chunk.id, () => extract(chunk.text));
	const seconds = (performance.now() - started) / 1000;
	chunks.push({ ...outcome, seconds });
	const result = outcome.ok
		? (outcome.result as { entities: unknown[]; relations: unknown[] })
		: null;
	console.log(
		`relex-ts ${index + 1}/${texts.chunks.length}: ` +
			(result
				? `${result.entities.length} entities, ${result.relations.length} relations`
				: outcome.error) +
			` (${seconds.toFixed(2)}s)`
	);
}

const edgeCases = [
	await attempt('no-entities', () => extract(EMPTY_TEXT)),
	await attempt('batch-of-two', () =>
		Promise.all([extract(texts.chunks[1].text), extract(texts.chunks[2].text)])
	)
];
for (const edge of edgeCases)
	console.log(`relex-ts edge case ${edge.case}: ${edge.ok ? 'ok' : edge.error}`);

writeFileSync(
	`${OUT}/relex-ts.json`,
	JSON.stringify(
		{
			variant: 'ts',
			threads: null,
			loadSeconds,
			peakRssMb: process.resourceUsage().maxRSS / 1024,
			chunks,
			edgeCases
		},
		null,
		2
	) + '\n'
);
