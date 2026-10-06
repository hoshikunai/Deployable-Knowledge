import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { eq, inArray } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { documentChunks, documents } from '../../src/lib/server/database/schema';
import { extractRelex, loadRelexRuntime } from '../../src/lib/server/entities/relex';
import { evidence } from './evidence.mts';

interface ProposedType {
	name: string;
	definition: string;
	head?: string;
	tail?: string;
}

interface ProposalFile {
	prompt?: string;
	proposal: { entityTypes: ProposedType[]; relationTypes: ProposedType[] };
	sampleChunkIds: string[];
}

interface LabelSet {
	entityLabels: string[];
	relationLabels: string[];
}

interface Candidate {
	name: string;
	definition: string;
	head?: string;
	tail?: string;
	proposedBy: string[];
	hits: number;
	documents: string[];
	examples: CandidateExample[];
}

interface CandidateExample {
	document: string;
	text: string;
	score: number;
	evidence?: string;
}

const { values } = parseArgs({
	options: {
		proposals: {
			type: 'string',
			default: '.cache/relex-eval/llm-labels-step1.json,.cache/relex-eval/llm-labels-step2.json'
		},
		labels: { type: 'string' },
		examples: { type: 'string', default: '4' },
		out: { type: 'string', default: '.cache/relex-eval/candidate-labels.json' }
	}
});

const EXAMPLE_COUNT = Number(values.examples);

function mergeTypes(files: ProposalFile[], kind: 'entityTypes' | 'relationTypes'): Candidate[] {
	const byName = new Map<string, Candidate>();
	for (const file of files) {
		const proposer = file.prompt ?? 'basic';
		for (const type of file.proposal[kind]) {
			const existing = byName.get(type.name);
			if (existing) {
				if (!existing.proposedBy.includes(proposer)) existing.proposedBy.push(proposer);
				continue;
			}
			byName.set(type.name, {
				name: type.name,
				definition: type.definition,
				...(kind === 'relationTypes' ? { head: type.head, tail: type.tail } : {}),
				proposedBy: [proposer],
				hits: 0,
				documents: [],
				examples: []
			});
		}
	}
	return [...byName.values()];
}

function addExample(candidate: Candidate, example: CandidateExample) {
	if (!candidate.documents.includes(example.document)) candidate.documents.push(example.document);
	const sameText = candidate.examples.find((existing) => existing.text === example.text);
	if (sameText) {
		if (example.score > sameText.score) Object.assign(sameText, example);
		return;
	}
	candidate.examples.push(example);
}

function pickExamples(examples: CandidateExample[]): CandidateExample[] {
	const sorted = [...examples].sort((a, b) => b.score - a.score);
	const picked: CandidateExample[] = [];
	for (const example of sorted) {
		if (picked.length === EXAMPLE_COUNT) break;
		if (!picked.some((existing) => existing.document === example.document)) picked.push(example);
	}
	for (const example of sorted) {
		if (picked.length === EXAMPLE_COUNT) break;
		if (!picked.includes(example)) picked.push(example);
	}
	return picked;
}

const files = values.proposals
	.split(',')
	.map((path) => JSON.parse(readFileSync(path, 'utf8')) as ProposalFile);
const chunkIds = [...new Set(files.flatMap((file) => file.sampleChunkIds))];
const entityCandidates = mergeTypes(files, 'entityTypes');
const relationCandidates = mergeTypes(files, 'relationTypes');
const entityByName = new Map(entityCandidates.map((candidate) => [candidate.name, candidate]));
const relationByName = new Map(relationCandidates.map((candidate) => [candidate.name, candidate]));

const chunkRows = await db
	.select({ id: documentChunks.id, content: documentChunks.content, title: documents.title })
	.from(documentChunks)
	.innerJoin(documents, eq(documents.id, documentChunks.documentId))
	.where(inArray(documentChunks.id, chunkIds));
console.log(
	`${entityCandidates.length} entity and ${relationCandidates.length} relation candidates; ` +
		`hit test on ${chunkRows.length} sample chunks.`
);

async function runPass(labels: LabelSet) {
	for (const chunk of chunkRows) {
		const result = await extractRelex(chunk.content, labels);
		const entityLabels = new Set<string>();
		for (const entity of result.entities) {
			const candidate = entityByName.get(entity.label);
			if (!candidate) continue;
			entityLabels.add(entity.label);
			addExample(candidate, { document: chunk.title, text: entity.text, score: entity.score });
		}
		for (const label of entityLabels) entityByName.get(label)!.hits += 1;
		const hitLabels = new Set<string>();
		for (const relation of result.relations) {
			const candidate = relationByName.get(relation.relation);
			if (!candidate) continue;
			hitLabels.add(relation.relation);
			addExample(candidate, {
				document: chunk.title,
				text: `${relation.head.text} → ${relation.tail.text}`,
				score: relation.score,
				evidence: evidence(chunk.content, relation.head, relation.tail)
			});
		}
		for (const label of hitLabels) relationByName.get(label)!.hits += 1;
	}
}

const labelSets: LabelSet[] = values.labels
	? [JSON.parse(readFileSync(values.labels, 'utf8')) as LabelSet]
	: files.map((file) => ({
			entityLabels: file.proposal.entityTypes.map((type) => type.name),
			relationLabels: file.proposal.relationTypes.map((type) => type.name)
		}));
const unknown = labelSets
	.flatMap((set) => [...set.entityLabels, ...set.relationLabels])
	.filter((name) => !entityByName.has(name) && !relationByName.has(name));
if (unknown.length) throw new Error(`Labels not in any proposal: ${unknown.join(', ')}`);

await loadRelexRuntime();
for (const [index, labels] of labelSets.entries()) {
	await runPass(labels);
	console.log(
		`Set ${index + 1}/${labelSets.length}: ${labels.entityLabels.length} entity and ` +
			`${labels.relationLabels.length} relation labels done.`
	);
}
const inSets = new Set(labelSets.flatMap((set) => [...set.entityLabels, ...set.relationLabels]));

const finish = (candidates: Candidate[]) =>
	candidates
		.filter((candidate) => inSets.has(candidate.name))
		.map((candidate) => ({
			...candidate,
			hitShare: candidate.hits / chunkRows.length,
			totalExamples: candidate.examples.length,
			examples: pickExamples(candidate.examples)
		}))
		.sort((a, b) => b.hits - a.hits);

const output = {
	labelSets,
	sampleChunks: chunkRows.length,
	documents: [...new Set(chunkRows.map((chunk) => chunk.title))],
	entityCandidates: finish(entityCandidates),
	relationCandidates: finish(relationCandidates)
};
writeFileSync(values.out, JSON.stringify(output, null, 2) + '\n');
for (const candidate of output.relationCandidates) {
	console.log(
		`${candidate.name}: ${candidate.hits}/${chunkRows.length} chunks, ` +
			`${candidate.totalExamples} relations, ${candidate.documents.length} documents`
	);
}
console.log(`Done: ${values.out}`);
process.exit(0);
