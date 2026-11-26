import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { inArray } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { documentChunks } from '../../src/lib/server/database/schema';
import {
	activeChatModel,
	releaseChatModel,
	countHits,
	GENERIC_NAMES,
	generateValidated,
	isVagueRelation,
	normalizeLabel,
	parseJsonObject,
	resolveSource,
	type LabelSet,
	type ProposedType
} from './label-tools.mts';

interface DictionaryEntry {
	title: string;
	domain?: string;
	sampleChunkIds: string[];
	entityTypes?: ProposedType[];
	relationTypes?: ProposedType[];
}

interface MergedType {
	name: string;
	definition: string;
	head?: string;
	tail?: string;
	mergedFrom: string[];
}

interface Merge {
	entityTypes: MergedType[];
	relationTypes: MergedType[];
}

interface Removal {
	name: string;
	kind: 'entity' | 'relation';
	reason: string;
}

const { values } = parseArgs({
	options: {
		model: { type: 'string' },
		dictionary: { type: 'string', default: '.cache/relex-eval/dictionary.json' },
		'max-share': { type: 'string', default: '0.8' },
		rounds: { type: 'string', default: '4' },
		out: { type: 'string', default: '.cache/relex-eval/labels-pipeline.json' }
	}
});

const MAX_LABELS = 10;
const MAX_TOKENS = 3000;
const MAX_SHARE = Number(values['max-share']);
const MAX_ROUNDS = Number(values.rounds);
const mergedTypeSchema = (withEnds: boolean) => ({
	type: 'object',
	properties: {
		name: { type: 'string' },
		definition: { type: 'string' },
		...(withEnds ? { head: { type: 'string' }, tail: { type: 'string' } } : {}),
		mergedFrom: { type: 'array', minItems: 1, items: { type: 'string' } }
	},
	required: ['name', 'definition', ...(withEnds ? ['head', 'tail'] : []), 'mergedFrom']
});
const RESPONSE_SCHEMA = {
	type: 'object',
	properties: {
		entityTypes: {
			type: 'array',
			minItems: 2,
			maxItems: MAX_LABELS,
			items: mergedTypeSchema(false)
		},
		relationTypes: {
			type: 'array',
			minItems: 2,
			maxItems: MAX_LABELS,
			items: mergedTypeSchema(true)
		}
	},
	required: ['entityTypes', 'relationTypes']
};

function describeEntry(entry: DictionaryEntry, position: number): string {
	const entities = (entry.entityTypes ?? [])
		.map((type) => `${type.name} (${type.definition})`)
		.join('; ');
	const relations = (entry.relationTypes ?? [])
		.map((type) => `${type.name}: ${type.head ?? '?'} -> ${type.tail ?? '?'} (${type.definition})`)
		.join('; ');
	return `[${position}] "${entry.title}": ${entry.domain ?? ''}
Entity types: ${entities}
Relation types: ${relations}`;
}

function buildPrompt(entries: DictionaryEntry[]): string {
	return `You are merging per-document label dictionaries into one label set for a relation extraction model that will read every chunk of a library of ${entries.length} documents. Labels compete inside the model's prompt, so the final set must be small: at most ${MAX_LABELS} entity types and at most ${MAX_LABELS} relation types.

Per-document dictionaries:
${entries.map((entry, index) => describeEntry(entry, index + 1)).join('\n\n')}

Rules:
- Merge synonyms and inverse pairs into one name read in one direction ("governs" and "is governed by" become one).
- Merge entity types that overlap ("unit" and "military unit" become one) and keep the more general name.
- Prefer names that several documents share, so the graph connects across documents.
- Drop relation types that come from layout rather than meaning (a document and its date or sections, items in the same list) and vague ones ("is located in", "contains", "applies to", "is related to", "is part of").
- Each type lists in "mergedFrom" the exact original names it replaces, copied from the dictionaries above.

Answer with JSON: "entityTypes" is a list of {"name", "definition", "mergedFrom"}; "relationTypes" is a list of {"name", "definition", "head", "tail", "mergedFrom"}.`;
}

function readMerged(value: unknown, kind: string): MergedType[] {
	if (!Array.isArray(value)) throw new Error(`'${kind}' is not a list.`);
	return value.map((item) => {
		if (!item || typeof item !== 'object')
			throw new Error(`An entry in '${kind}' is not an object.`);
		const record = item as Record<string, unknown>;
		if (typeof record.name !== 'string' || !Array.isArray(record.mergedFrom)) {
			throw new Error(`An entry in '${kind}' has no name or mergedFrom list.`);
		}
		return {
			name: normalizeLabel(record.name),
			definition: typeof record.definition === 'string' ? record.definition : '',
			...(typeof record.head === 'string' ? { head: record.head } : {}),
			...(typeof record.tail === 'string' ? { tail: record.tail } : {}),
			mergedFrom: record.mergedFrom
				.filter((source): source is string => typeof source === 'string')
				.map(normalizeLabel)
		};
	});
}

function parseMerge(output: string): Merge {
	const parsed = parseJsonObject(output);
	return {
		entityTypes: readMerged(parsed.entityTypes, 'entityTypes'),
		relationTypes: readMerged(parsed.relationTypes, 'relationTypes')
	};
}

function stem(token: string): string {
	return token.length > 3 && token.endsWith('s') && !token.endsWith('ss')
		? token.slice(0, -1)
		: token;
}

function tokens(name: string): Set<string> {
	return new Set(name.split(' ').map(stem));
}

function isSubset(small: Set<string>, large: Set<string>): boolean {
	return [...small].every((token) => large.has(token));
}

function foldOverlappingEntities(types: MergedType[], removals: Removal[]): MergedType[] {
	const kept = [...types].sort((a, b) => a.name.split(' ').length - b.name.split(' ').length);
	const result: MergedType[] = [];
	for (const type of kept) {
		const general = result.find((existing) => isSubset(tokens(existing.name), tokens(type.name)));
		if (!general) {
			result.push({ ...type, mergedFrom: [...type.mergedFrom] });
			continue;
		}
		general.mergedFrom.push(type.name, ...type.mergedFrom);
		removals.push({
			name: type.name,
			kind: 'entity',
			reason: `folded into "${general.name}"`
		});
	}
	return result;
}

function filterMerged(merge: Merge, removals: Removal[]): Merge {
	const keep = (kind: Removal['kind']) => (type: MergedType) => {
		const reason =
			(GENERIC_NAMES.has(type.name) && 'generic name') ||
			(kind === 'relation' && isVagueRelation(type.name) && 'vague or layout relation') ||
			'';
		if (reason) removals.push({ name: type.name, kind, reason });
		return !reason;
	};
	const unique = (types: MergedType[]) =>
		types.filter((type, index) => types.findIndex((other) => other.name === type.name) === index);
	return {
		entityTypes: foldOverlappingEntities(
			unique(merge.entityTypes.filter(keep('entity'))),
			removals
		),
		relationTypes: unique(merge.relationTypes.filter(keep('relation')))
	};
}

async function hitTest(texts: string[], start: LabelSet, removals: Removal[]) {
	const labels = {
		entityLabels: [...start.entityLabels],
		relationLabels: [...start.relationLabels]
	};
	const rounds = [];
	for (let round = 1; round <= MAX_ROUNDS; round += 1) {
		const { entityHits, relationHits, relations } = await countHits(texts, labels);
		rounds.push({
			labels: structuredClone(labels),
			relations,
			entityHits: Object.fromEntries(entityHits),
			relationHits: Object.fromEntries(relationHits)
		});
		const drop = (hits: Map<string, number>, kind: Removal['kind']) =>
			[...hits].flatMap(([name, count]) => {
				if (count === 0) return [{ name, kind, reason: `0 hits in round ${round}` }];
				if (count > texts.length * MAX_SHARE) {
					return [{ name, kind, reason: `${count}/${texts.length} chunks in round ${round}` }];
				}
				return [];
			});
		const dropped = [...drop(entityHits, 'entity'), ...drop(relationHits, 'relation')];
		console.log(
			`Round ${round}: ${labels.entityLabels.length} entity, ${labels.relationLabels.length} relation labels, ` +
				`${relations} relations, dropping ${dropped.map((item) => item.name).join(', ') || 'nothing'}`
		);
		if (!dropped.length) break;
		removals.push(...dropped);
		const names = new Set(dropped.map((item) => item.name));
		labels.entityLabels = labels.entityLabels.filter((name) => !names.has(name));
		labels.relationLabels = labels.relationLabels.filter((name) => !names.has(name));
		if (labels.entityLabels.length < 2 || labels.relationLabels.length < 2) {
			throw new Error(
				`The hit test left ${labels.entityLabels.length} entity and ${labels.relationLabels.length} relation labels.`
			);
		}
	}
	return { labels, rounds };
}

function perDocumentLabels(entries: DictionaryEntry[], merge: Merge, labels: LabelSet) {
	const owners = (types: MergedType[], survivors: string[]) => {
		const owner = new Map<string, string>();
		for (const type of types) {
			if (!survivors.includes(type.name)) continue;
			for (const source of [type.name, ...type.mergedFrom]) owner.set(source, type.name);
		}
		return owner;
	};
	const entityOwner = owners(merge.entityTypes, labels.entityLabels);
	const relationOwner = owners(merge.relationTypes, labels.relationLabels);
	const map = (names: string[], owner: Map<string, string>) => [
		...new Set(names.flatMap((name) => owner.get(name) ?? []))
	];
	return Object.fromEntries(
		entries.map((entry) => [
			entry.title,
			{
				entityLabels: map(
					(entry.entityTypes ?? []).map((type) => type.name),
					entityOwner
				),
				relationLabels: map(
					(entry.relationTypes ?? []).map((type) => type.name),
					relationOwner
				)
			}
		])
	);
}

function resolveSources(merge: Merge, originalNames: string[], unmatched: string[]): Merge {
	const resolveType = (type: MergedType): MergedType => ({
		...type,
		mergedFrom: type.mergedFrom.flatMap((source) => {
			const name = resolveSource(source, originalNames);
			if (!name) unmatched.push(`${type.name} <- ${source}`);
			return name ? [name] : [];
		})
	});
	return {
		entityTypes: merge.entityTypes.map(resolveType),
		relationTypes: merge.relationTypes.map(resolveType)
	};
}

const dictionary = JSON.parse(readFileSync(values.dictionary, 'utf8')) as {
	entries: DictionaryEntry[];
};
const entries = dictionary.entries.filter((entry) => entry.entityTypes && entry.relationTypes);
const failed = dictionary.entries.length - entries.length;
console.log(`${entries.length} dictionary entries${failed ? `, ${failed} failed in pass 1` : ''}.`);

const model = await activeChatModel(values.model);
const { value: proposed, attempts } = await generateValidated(
	model,
	buildPrompt(entries),
	RESPONSE_SCHEMA,
	MAX_TOKENS,
	parseMerge
);
await releaseChatModel(model);
for (const [index, attempt] of attempts.entries()) {
	console.log(
		`Merge attempt ${index + 1}: ${attempt.error ? `failed: ${attempt.error}` : 'valid'} after ${attempt.seconds.toFixed(1)}s.`
	);
}
if (!proposed) throw new Error('The model did not return a usable merge in two attempts.');

const originalNames = [
	...new Set(
		entries.flatMap((entry) =>
			[...(entry.entityTypes ?? []), ...(entry.relationTypes ?? [])].map((type) => type.name)
		)
	)
].sort((a, b) => b.length - a.length);
const unmatchedSources: string[] = [];
const resolved = resolveSources(proposed, originalNames, unmatchedSources);
const removals: Removal[] = [];
const filtered = filterMerged(resolved, removals);
const startLabels = {
	entityLabels: filtered.entityTypes.map((type) => type.name),
	relationLabels: filtered.relationTypes.map((type) => type.name)
};
console.log(
	`After filters: ${startLabels.entityLabels.join(', ')} | ${startLabels.relationLabels.join(', ')}`
);

const chunkIds = entries.flatMap((entry) => entry.sampleChunkIds);
const chunkRows = await db
	.select({ content: documentChunks.content })
	.from(documentChunks)
	.where(inArray(documentChunks.id, chunkIds));
const { labels, rounds } = await hitTest(
	chunkRows.map((row) => row.content),
	startLabels,
	removals
);
const perDocument = perDocumentLabels(entries, filtered, labels);

writeFileSync(values.out, JSON.stringify(labels, null, 2) + '\n');
writeFileSync(
	values.out.replace(/\.json$/, '-per-document.json'),
	JSON.stringify(perDocument, null, 2) + '\n'
);
writeFileSync(
	values.out.replace(/\.json$/, '-merge.json'),
	JSON.stringify(
		{
			proposed,
			resolved,
			filtered,
			unmatchedSources,
			removals,
			rounds,
			labels,
			perDocument,
			attempts
		},
		null,
		2
	) + '\n'
);
console.log(JSON.stringify({ labels, removals, unmatchedSources, perDocument }, null, 2));
console.log(`Done: ${values.out}`);
process.exit(0);
