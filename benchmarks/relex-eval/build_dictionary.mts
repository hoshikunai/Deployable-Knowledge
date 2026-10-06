import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { inArray } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { documentChunks, documents } from '../../src/lib/server/database/schema';
import { disposeRuntime } from '../../src/lib/server/providers/llamacpp-runtime';
import { getVectorIndex } from '../../src/lib/server/rag/search/vector-index';
import {
	activeChatModel,
	entityTypeSchema,
	generateValidated,
	normalizeText,
	parseJsonObject,
	readTypes,
	relationTypeSchema,
	representativeRows,
	seededRandom,
	type GenerationAttempt,
	type ProposedType,
	type ReadContext
} from './label-tools.mts';

interface DocumentEntry {
	domain: string;
	entityTypes: ProposedType[];
	relationTypes: ProposedType[];
}

interface DictionaryEntry extends Partial<DocumentEntry> {
	documentId: string;
	title: string;
	sampleChunkIds: string[];
	reused?: { entity: string[]; relation: string[] };
	rejectedAsVague?: string[];
	ungrounded?: string[];
	attempts: Omit<GenerationAttempt, 'output'>[];
	error?: string;
}

const { values } = parseArgs({
	options: {
		'per-document': { type: 'string', default: '8' },
		seed: { type: 'string', default: '42' },
		exclude: { type: 'string', default: '.cache/relex-eval/own-chunks.json' },
		out: { type: 'string', default: '.cache/relex-eval/dictionary.json' }
	}
});

const PER_DOCUMENT = Number(values['per-document']);
const EXCERPT_CHARACTERS = 600;
const MAX_LABELS = 8;
const MAX_TOKENS = 3000;
const RESPONSE_SCHEMA = {
	type: 'object',
	properties: {
		domain: { type: 'string' },
		entityTypes: { type: 'array', minItems: 2, maxItems: MAX_LABELS, items: entityTypeSchema },
		relationTypes: { type: 'array', minItems: 1, maxItems: MAX_LABELS, items: relationTypeSchema }
	},
	required: ['domain', 'entityTypes', 'relationTypes']
};

function knownLabels(entries: DictionaryEntry[]): string {
	const entityNames = [
		...new Set(entries.flatMap((entry) => (entry.entityTypes ?? []).map((type) => type.name)))
	];
	const relations = new Map<string, ProposedType>();
	for (const type of entries.flatMap((entry) => entry.relationTypes ?? [])) {
		if (!relations.has(type.name)) relations.set(type.name, type);
	}
	if (!entityNames.length && !relations.size) {
		return 'This is the first document, so the dictionary is still empty.';
	}
	const relationLines = [...relations.values()].map(
		(type) => `${type.name} (${type.head ?? '?'} -> ${type.tail ?? '?'})`
	);
	return `Labels already in the dictionary from earlier documents. Reuse a name exactly when it means the same thing here; propose a new name only for something they do not cover.
Entity types: ${entityNames.join(', ')}
Relation types: ${relationLines.join('; ')}`;
}

function buildPrompt(
	title: string,
	position: number,
	total: number,
	excerpts: string[],
	known: string
): string {
	return `You are building a label dictionary for a relation extraction model, one document at a time. This is document ${position} of ${total}: "${title}". Below are ${excerpts.length} representative excerpts from it.

${known}

Describe this document in one line ("domain"), then propose up to ${MAX_LABELS} entity types and up to ${MAX_LABELS} relation types that occur in it.

Entity types:
- Reusable categories such as "medication", "company" or "job role", never specific names.
- Not generic: object, thing, item, entity, concept, other, information, data.
- Not document layout: page headers, publication numbers, section titles, figure captions, issue dates.

Relation types:
- A specific verb phrase that a sentence in the excerpts states outright, read from head to tail. Good: "treats" (medication -> condition), "is manufactured by" (product -> company), "reports to" (job role -> job role), "was founded by" (organization -> person).
- It must say how two things are related, not only that they are. Bad: "is located in", "contains", "includes", "has", "is part of", "applies to", "is related to", "is associated with", "is classified as", "involves", "requires equipment", "is dated on", "is detailed in".
- Never a relation that comes from layout rather than meaning: a document and its date, a document and its sections, two items in the same bulleted list.
- Never a relation that the text negates ("do not", "is not authorized").
- For each relation type give one example from the excerpts: the head, the tail, and the sentence copied word for word.

Answer with JSON: "domain" is one line; "entityTypes" is a list of {"name", "definition"}; "relationTypes" is a list of {"name", "definition", "head", "tail", "example": {"head", "tail", "sentence"}}.

Excerpts:
${excerpts.map((excerpt, index) => `[${index + 1}] ${excerpt}`).join('\n\n')}`;
}

function parseEntry(output: string, context: ReadContext): DocumentEntry {
	const parsed = parseJsonObject(output);
	context.rejected.length = 0;
	if (typeof parsed.domain !== 'string') throw new Error("The reply has no 'domain' line.");
	const entry = {
		domain: parsed.domain.trim(),
		entityTypes: readTypes(parsed.entityTypes, 'entityTypes', context),
		relationTypes: readTypes(parsed.relationTypes, 'relationTypes', context)
	};
	if (entry.entityTypes.length < 2 || entry.relationTypes.length < 1) {
		throw new Error('The reply has fewer than 2 usable entity types or 1 relation type.');
	}
	return entry;
}

const excluded = new Set(
	(JSON.parse(readFileSync(values.exclude, 'utf8')) as { id: string }[]).map((chunk) => chunk.id)
);
const model = await activeChatModel();
console.log(
	`Active profile: ${model.profile.name} · ${model.profile.provider} · ${model.profile.model}`
);

const index = await getVectorIndex();
const random = seededRandom(Number(values.seed));
const rowsByDocument = new Map<string, number[]>();
for (let row = 0; row < index.count; row += 1) {
	if (index.chunkTypes[row] !== 'TEXT' || excluded.has(index.chunkIds[row])) continue;
	const documentRows = rowsByDocument.get(index.documentIds[row]);
	if (documentRows) documentRows.push(row);
	else rowsByDocument.set(index.documentIds[row], [row]);
}
const titleRows = await db
	.select({ id: documents.id, title: documents.title })
	.from(documents)
	.where(inArray(documents.id, [...rowsByDocument.keys()]));
const titleById = new Map(titleRows.map((row) => [row.id, row.title]));
console.log(
	`${rowsByDocument.size} documents, ${PER_DOCUMENT} excerpts each, ${excluded.size} test chunks excluded.`
);

const entries: DictionaryEntry[] = [];
const rawAttempts: Record<string, GenerationAttempt[]> = {};
let position = 0;
for (const [documentId, documentRows] of rowsByDocument) {
	position += 1;
	const title = titleById.get(documentId) ?? documentId;
	const rows = representativeRows(
		index.matrix,
		documentRows,
		index.dimensions,
		Math.min(PER_DOCUMENT, documentRows.length),
		random
	);
	const chunkIds = rows.map((row) => index.chunkIds[row]);
	const chunkRows = await db
		.select({ id: documentChunks.id, content: documentChunks.content })
		.from(documentChunks)
		.where(inArray(documentChunks.id, chunkIds));
	const contentById = new Map(chunkRows.map((row) => [row.id, row.content]));
	const sampleChunkIds = chunkIds.filter((id) => contentById.has(id));
	const excerpts = sampleChunkIds.map((id) =>
		contentById.get(id)!.replace(/\s+/g, ' ').slice(0, EXCERPT_CHARACTERS)
	);
	const context: ReadContext = {
		excerptText: normalizeText(excerpts.join(' ')),
		rejected: [],
		maxLabels: MAX_LABELS
	};
	const prompt = buildPrompt(title, position, rowsByDocument.size, excerpts, knownLabels(entries));
	const { value, attempts } = await generateValidated(
		model,
		prompt,
		RESPONSE_SCHEMA,
		MAX_TOKENS,
		(output) => parseEntry(output, context)
	);
	rawAttempts[title] = attempts;
	const knownEntity = new Set(
		entries.flatMap((entry) => entry.entityTypes ?? []).map((type) => type.name)
	);
	const knownRelation = new Set(
		entries.flatMap((entry) => entry.relationTypes ?? []).map((type) => type.name)
	);
	const entry: DictionaryEntry = {
		documentId,
		title,
		sampleChunkIds,
		attempts: attempts.map(({ seconds, error }) => ({ seconds, ...(error ? { error } : {}) }))
	};
	if (value) {
		Object.assign(entry, value, {
			reused: {
				entity: value.entityTypes
					.filter((type) => knownEntity.has(type.name))
					.map((type) => type.name),
				relation: value.relationTypes
					.filter((type) => knownRelation.has(type.name))
					.map((type) => type.name)
			},
			rejectedAsVague: [...context.rejected],
			ungrounded: value.relationTypes
				.filter((type) => type.grounded === false)
				.map((type) => type.name)
		});
	} else {
		entry.error = attempts.at(-1)?.error ?? 'No reply.';
	}
	entries.push(entry);
	writeFileSync(values.out, JSON.stringify({ perDocument: PER_DOCUMENT, entries }, null, 2) + '\n');
	writeFileSync(
		values.out.replace(/\.json$/, '-attempts.json'),
		JSON.stringify(rawAttempts, null, 2) + '\n'
	);
	const seconds = attempts.reduce((total, attempt) => total + attempt.seconds, 0);
	console.log(
		value
			? `${position}/${rowsByDocument.size} ${title}: ${value.entityTypes.length} entity, ` +
					`${value.relationTypes.length} relation types (${entry.reused!.relation.length} reused) in ${seconds.toFixed(0)}s`
			: `${position}/${rowsByDocument.size} ${title}: FAILED after ${seconds.toFixed(0)}s: ${entry.error}`
	);
}

await disposeRuntime();
console.log(`Done: ${values.out}`);
process.exit(0);
