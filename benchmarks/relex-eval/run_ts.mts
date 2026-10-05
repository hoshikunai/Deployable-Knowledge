import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { DEFAULT_RELEX_THRESHOLDS } from '../../src/lib/constants/entity-extraction';
import { extractRelex, loadRelexRuntime } from '../../src/lib/server/entities/relex';

interface EvaluationItem {
	id: string;
	paragraphs: { title: string; text: string }[];
}

interface EvaluationSet {
	entityLabels: string[];
	relationLabels: string[];
	items: EvaluationItem[];
}

const { values } = parseArgs({
	options: {
		set: { type: 'string', default: '.cache/relex-eval/2wiki-set.json' },
		out: { type: 'string', default: '.cache/relex-eval/2wiki-ts.jsonl' },
		'relation-floor': { type: 'string', default: '0.3' },
		limit: { type: 'string' }
	}
});

const relationFloor = Number(values['relation-floor']);
if (!(relationFloor > 0 && relationFloor < 1))
	throw new Error('--relation-floor must be between 0 and 1.');
const evaluationSet = JSON.parse(readFileSync(values.set, 'utf8')) as EvaluationSet;
const limit = values.limit ? Number(values.limit) : evaluationSet.items.length;
const items = evaluationSet.items.slice(0, limit);
const done = new Set(
	existsSync(values.out)
		? readFileSync(values.out, 'utf8')
				.split('\n')
				.filter(Boolean)
				.map((line) => (JSON.parse(line) as { id: string }).id)
		: []
);
const thresholds = { ...DEFAULT_RELEX_THRESHOLDS, relation: relationFloor };

const loadStarted = performance.now();
await loadRelexRuntime();
console.log(
	`Model loaded in ${((performance.now() - loadStarted) / 1000).toFixed(1)}s; ` +
		`${done.size} of ${items.length} questions already done; thresholds ${JSON.stringify(thresholds)}.`
);

const runStarted = performance.now();
let processed = 0;
for (const [index, item] of items.entries()) {
	if (done.has(item.id)) continue;
	const started = performance.now();
	const paragraphs = [];
	for (const paragraph of item.paragraphs) {
		const result = await extractRelex(paragraph.text, {
			entityLabels: evaluationSet.entityLabels,
			relationLabels: evaluationSet.relationLabels,
			thresholds
		});
		paragraphs.push({
			title: paragraph.title,
			entities: result.entities.map(({ text, label, score }) => ({ text, label, score })),
			relations: result.relations.map((relation) => ({
				head: relation.head.text,
				relation: relation.relation,
				tail: relation.tail.text,
				score: relation.score
			}))
		});
	}
	const seconds = (performance.now() - started) / 1000;
	appendFileSync(
		values.out,
		JSON.stringify({ id: item.id, seconds, thresholds, paragraphs }) + '\n'
	);
	processed += 1;
	if (processed % 25 === 0 || index === items.length - 1) {
		const elapsed = (performance.now() - runStarted) / 1000;
		console.log(
			`${index + 1}/${items.length} questions, ${(elapsed / processed).toFixed(2)}s per question`
		);
	}
}
console.log(`Done: ${values.out}`);
