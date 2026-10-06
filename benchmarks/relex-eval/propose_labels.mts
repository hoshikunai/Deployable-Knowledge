import { writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { inArray } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { documentChunks } from '../../src/lib/server/database/schema';
import { disposeRuntime } from '../../src/lib/server/providers/llamacpp-runtime';
import { getVectorIndex } from '../../src/lib/server/rag/search/vector-index';
import {
	activeChatModel,
	countHits,
	entityTypeSchema,
	generateValidated,
	normalizeText,
	parseJsonObject,
	readTypes,
	relationTypeSchema,
	representativeRows,
	seededRandom,
	type ProposedType,
	type ReadContext
} from './label-tools.mts';

interface Proposal {
	entityTypes: ProposedType[];
	relationTypes: ProposedType[];
}

const { values } = parseArgs({
	options: {
		samples: { type: 'string', default: '20' },
		sampling: { type: 'string', default: 'document' },
		'per-document': { type: 'string', default: '2' },
		prompt: { type: 'string', default: 'specific' },
		seed: { type: 'string', default: '42' },
		out: { type: 'string', default: '.cache/relex-eval/llm-labels.json' }
	}
});

const SAMPLE_COUNT = Number(values.samples);
const PER_DOCUMENT = Number(values['per-document']);
if (values.sampling !== 'document' && values.sampling !== 'cluster') {
	throw new Error("--sampling must be 'document' or 'cluster'.");
}
if (values.prompt !== 'basic' && values.prompt !== 'specific') {
	throw new Error("--prompt must be 'basic' or 'specific'.");
}
const PROMPT_STYLE = values.prompt;
const EXCERPT_CHARACTERS = 600;
const MAX_LABELS = 10;
const MAX_TOKENS = 3000;
const RESPONSE_SCHEMA = {
	type: 'object',
	properties: {
		entityTypes: {
			type: 'array',
			minItems: 3,
			maxItems: MAX_LABELS,
			items: entityTypeSchema
		},
		relationTypes: {
			type: 'array',
			minItems: 2,
			maxItems: MAX_LABELS,
			items:
				PROMPT_STYLE === 'specific'
					? relationTypeSchema
					: {
							type: 'object',
							properties: {
								name: { type: 'string' },
								definition: { type: 'string' },
								head: { type: 'string' },
								tail: { type: 'string' }
							},
							required: ['name', 'definition', 'head', 'tail']
						}
		}
	},
	required: ['entityTypes', 'relationTypes']
};

const PROMPT_RULES = {
	basic: `Propose up to ${MAX_LABELS} entity types and up to ${MAX_LABELS} relation types that occur in these documents.
Rules:
- Entity types are reusable categories such as "medication", never specific names.
- Do not use generic types such as object, thing, item, entity, concept, other, information or data.
- Relation types are short verb phrases read from head to tail, such as "treats" (medication -> injury).
- Give each a one-line definition. For each relation type, name the entity types of its head and tail.

Answer with JSON: "entityTypes" is a list of {"name", "definition"}; "relationTypes" is a list of {"name", "definition", "head", "tail"}.`,
	specific: `A relation extraction model will read every chunk of this library with your labels and link pairs of entities. Vague labels make it link everything to everything, so every label must be specific.

Propose up to ${MAX_LABELS} entity types and up to ${MAX_LABELS} relation types.

Entity types:
- Reusable categories such as "medication", "aircraft system" or "rating official", never specific names.
- Not generic: object, thing, item, entity, concept, other, information, data.
- Not document layout: page headers, publication numbers, section titles, figure captions, issue dates.

Relation types:
- A specific verb phrase that a sentence in the excerpts states outright, read from head to tail. Good: "treats" (medication -> injury), "is rated by" (officer -> commander), "is approved by" (award -> authority), "is authorized for" (award -> service member).
- It must say how two things are related, not only that they are. Bad: "is located in", "contains", "includes", "has", "is part of", "applies to", "is related to", "is associated with", "is classified as", "involves", "requires equipment", "is dated on".
- Never a relation that comes from layout rather than meaning: a document and its date, a document and its sections, two items in the same bulleted list.
- Never a relation that the text negates ("do not", "is not authorized").
- For each relation type give one example from the excerpts: the head, the tail, and the sentence copied word for word.

Answer with JSON: "entityTypes" is a list of {"name", "definition"}; "relationTypes" is a list of {"name", "definition", "head", "tail", "example": {"head", "tail", "sentence"}}.`
};

function buildPrompt(excerpts: string[]): string {
	return `You are choosing extraction labels for a document library. Below are ${excerpts.length} representative excerpts from it.

${PROMPT_RULES[PROMPT_STYLE]}

Excerpts:
${excerpts.map((excerpt, index) => `[${index + 1}] ${excerpt}`).join('\n\n')}`;
}

function parseProposal(output: string, context: ReadContext): Proposal {
	const parsed = parseJsonObject(output);
	context.rejected.length = 0;
	const proposal = {
		entityTypes: readTypes(parsed.entityTypes, 'entityTypes', context),
		relationTypes: readTypes(parsed.relationTypes, 'relationTypes', context)
	};
	if (proposal.entityTypes.length < 3 || proposal.relationTypes.length < 2) {
		throw new Error('The reply has fewer than 3 usable entity types or 2 relation types.');
	}
	return proposal;
}

const model = await activeChatModel();
const { profile } = model;
console.log(`Active profile: ${profile.name} · ${profile.provider} · ${profile.model}`);

const index = await getVectorIndex();
const random = seededRandom(Number(values.seed));
const textRows = Array.from({ length: index.count }, (_, row) => row).filter(
	(row) => index.chunkTypes[row] === 'TEXT'
);
const rowsByDocument = new Map<string, number[]>();
for (const row of textRows) {
	const documentRows = rowsByDocument.get(index.documentIds[row]);
	if (documentRows) documentRows.push(row);
	else rowsByDocument.set(index.documentIds[row], [row]);
}
const rows =
	values.sampling === 'cluster'
		? representativeRows(index.matrix, textRows, index.dimensions, SAMPLE_COUNT, random)
		: [...rowsByDocument.values()].flatMap((documentRows) =>
				representativeRows(
					index.matrix,
					documentRows,
					index.dimensions,
					Math.min(PER_DOCUMENT, documentRows.length),
					random
				)
			);
const chunkIds = rows.map((row) => index.chunkIds[row]);
const chunkRows = await db
	.select({ id: documentChunks.id, content: documentChunks.content })
	.from(documentChunks)
	.where(inArray(documentChunks.id, chunkIds));
const contentById = new Map(chunkRows.map((row) => [row.id, row.content]));
const samples = chunkIds
	.map((id) => ({ id, text: contentById.get(id) ?? '' }))
	.filter((sample) => sample.text);
console.log(
	`Sampled ${samples.length} chunks (${values.sampling} sampling) from ${textRows.length} text chunks ` +
		`in ${rowsByDocument.size} documents.`
);

const excerpts = samples.map((sample) =>
	sample.text.replace(/\s+/g, ' ').slice(0, EXCERPT_CHARACTERS)
);
const context: ReadContext = {
	excerptText: normalizeText(excerpts.join(' ')),
	rejected: [],
	maxLabels: MAX_LABELS
};
const prompt = buildPrompt(excerpts);
const { value: proposal, attempts } = await generateValidated(
	model,
	prompt,
	RESPONSE_SCHEMA,
	MAX_TOKENS,
	(output) => parseProposal(output, context)
);
for (const [index, attempt] of attempts.entries()) {
	console.log(
		`Attempt ${index + 1}: ${attempt.error ? `failed: ${attempt.error}` : 'valid proposal'} ` +
			`after ${attempt.seconds.toFixed(1)}s.`
	);
}
await disposeRuntime();
writeFileSync(
	values.out.replace(/\.json$/, '-attempts.json'),
	JSON.stringify(attempts, null, 2) + '\n'
);
if (!proposal) throw new Error('The model did not return usable labels in two attempts.');
console.log(
	`Proposed ${proposal.entityTypes.length} entity and ${proposal.relationTypes.length} relation types.`
);

const entityNames = proposal.entityTypes.map((type) => type.name);
const relationNames = proposal.relationTypes.map((type) => type.name);
const { entityHits, relationHits } = await countHits(
	samples.map((sample) => sample.text),
	{ entityLabels: entityNames, relationLabels: relationNames }
);
const kept = (hits: Map<string, number>) =>
	[...hits].filter(([, count]) => count > 0).map(([name]) => name);
const flagged = (hits: Map<string, number>) =>
	[...hits].filter(([, count]) => count > samples.length * 0.8).map(([name]) => name);

const result = {
	model: `${profile.provider} · ${profile.model}`,
	sampling: values.sampling,
	prompt: PROMPT_STYLE,
	failedAttempts: attempts.filter((attempt) => attempt.error).length,
	entityLabels: kept(entityHits),
	relationLabels: kept(relationHits),
	dropped: {
		entity: entityNames.filter((name) => !entityHits.get(name)),
		relation: relationNames.filter((name) => !relationHits.get(name))
	},
	rejectedAsVague: context.rejected,
	ungrounded: proposal.relationTypes
		.filter((type) => type.grounded === false)
		.map((type) => type.name),
	flagged: { entity: flagged(entityHits), relation: flagged(relationHits) },
	hits: { entity: Object.fromEntries(entityHits), relation: Object.fromEntries(relationHits) },
	proposal,
	attempts,
	sampleChunkIds: samples.map((sample) => sample.id)
};
writeFileSync(values.out, JSON.stringify(result, null, 2) + '\n');
console.log(
	JSON.stringify(
		{
			entityLabels: result.entityLabels,
			relationLabels: result.relationLabels,
			dropped: result.dropped,
			rejectedAsVague: result.rejectedAsVague,
			ungrounded: result.ungrounded,
			flagged: result.flagged
		},
		null,
		2
	)
);
console.log(`Done: ${values.out}`);
process.exit(0);
