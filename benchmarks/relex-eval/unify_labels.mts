import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { inArray } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { documentChunks } from '../../src/lib/server/database/schema';
import { disposeRuntime } from '../../src/lib/server/providers/llamacpp-runtime';
import {
	activeChatModel,
	countHits,
	GENERIC_NAMES,
	generateValidated,
	isNegatedLabel,
	isVagueRelation,
	normalizeLabel,
	parseJsonObject,
	resolveSource,
	type LabelSet,
	type ProposedType
} from './label-tools.mts';

interface DictionaryEntry {
	title: string;
	sampleChunkIds: string[];
	entityTypes?: ProposedType[];
	relationTypes?: ProposedType[];
}

interface Group {
	name: string;
	members: string[];
}

interface Grouping {
	relationGroups: Group[];
}

interface Removal {
	document: string;
	name: string;
	kind: 'entity' | 'relation';
	reason: string;
}

const { values } = parseArgs({
	options: {
		dictionary: { type: 'string', default: '.cache/relex-eval/dictionary.json' },
		rounds: { type: 'string', default: '3' },
		out: { type: 'string', default: '.cache/relex-eval/labels-unified-per-document.json' }
	}
});

const MAX_TOKENS = 3000;
const MAX_ROUNDS = Number(values.rounds);
const groupSchema = {
	type: 'object',
	properties: {
		name: { type: 'string' },
		members: { type: 'array', minItems: 2, items: { type: 'string' } }
	},
	required: ['name', 'members']
};
const RESPONSE_SCHEMA = {
	type: 'object',
	properties: { relationGroups: { type: 'array', items: groupSchema } },
	required: ['relationGroups']
};

function labelList(types: ProposedType[]): string {
	const counts = new Map<string, number>();
	const definitions = new Map<string, string>();
	for (const type of types) {
		counts.set(type.name, (counts.get(type.name) ?? 0) + 1);
		if (!definitions.has(type.name)) definitions.set(type.name, type.definition);
	}
	return [...counts]
		.sort((a, b) => a[0].localeCompare(b[0]))
		.map(
			([name, count]) =>
				`- ${name} [${count} doc${count === 1 ? '' : 's'}]: ${definitions.get(name)}`
		)
		.join('\n');
}

function buildPrompt(entries: DictionaryEntry[]): string {
	const relations = entries.flatMap((entry) => entry.relationTypes ?? []);
	return `These relation labels were proposed separately for ${entries.length} documents of one library. The same relation sometimes got different names in different documents. Find those cases so the names can be unified.

Relation labels:
${labelList(relations)}

Rules:
- Group labels only when they mean the same thing, so that every fact one would extract the other would extract too: "is awarded to" and "may be awarded to", "is written by" and "is compiled by".
- An inverse pair is the same thing read the other way: "awards" and "is awarded to" form one group, named in one direction.
- Never generalize. Labels that would extract different facts stay apart, even when they are similar: "treats" and "is performed at" stay apart; "requires" and "requires submission to" stay apart.
- Name each group with one of its members where possible.
- Copy every member exactly as it is written in the list above, without the bracket or definition.
- Leave out labels that have no synonym. Fewer, surer groups are better than many loose ones.

Answer with JSON: "relationGroups" is a list of {"name", "members"}.`;
}

function readGroups(value: unknown, kind: string): Group[] {
	if (!Array.isArray(value)) throw new Error(`'${kind}' is not a list.`);
	return value.map((item) => {
		if (!item || typeof item !== 'object')
			throw new Error(`An entry in '${kind}' is not an object.`);
		const record = item as Record<string, unknown>;
		if (typeof record.name !== 'string' || !Array.isArray(record.members)) {
			throw new Error(`An entry in '${kind}' has no name or members.`);
		}
		return {
			name: normalizeLabel(record.name),
			members: record.members.filter((member): member is string => typeof member === 'string')
		};
	});
}

function parseGrouping(output: string): Grouping {
	const parsed = parseJsonObject(output);
	return { relationGroups: readGroups(parsed.relationGroups, 'relationGroups') };
}

function renames(groups: Group[], originalNames: string[], unmatched: string[]) {
	const rename = new Map<string, string>();
	for (const group of groups) {
		const members = group.members.flatMap((member) => {
			const name = resolveSource(member, originalNames);
			if (!name) unmatched.push(`${group.name} <- ${member}`);
			return name ? [name] : [];
		});
		if (new Set(members).size < 2) continue;
		for (const member of members) rename.set(member, group.name);
	}
	return rename;
}

function documentLabels(
	entry: DictionaryEntry,
	relationRename: Map<string, string>,
	removals: Removal[]
): LabelSet {
	const pick = (types: ProposedType[], rename: Map<string, string>, kind: Removal['kind']) => {
		const names = types.map((type) => rename.get(type.name) ?? type.name);
		return [...new Set(names)].filter((name) => {
			const reason =
				(GENERIC_NAMES.has(name) && 'generic name') ||
				(kind === 'relation' && isVagueRelation(name) && 'vague or layout relation') ||
				(kind === 'relation' && isNegatedLabel(name) && 'negated relation') ||
				'';
			if (reason) removals.push({ document: entry.title, name, kind, reason });
			return !reason;
		});
	};
	return {
		entityLabels: pick(entry.entityTypes ?? [], new Map(), 'entity'),
		relationLabels: pick(entry.relationTypes ?? [], relationRename, 'relation')
	};
}

async function hitTest(title: string, texts: string[], start: LabelSet, removals: Removal[]) {
	const labels = structuredClone(start);
	for (let round = 1; round <= MAX_ROUNDS; round += 1) {
		const { entityHits, relationHits } = await countHits(texts, labels);
		const unused = (hits: Map<string, number>) =>
			[...hits].filter(([, count]) => count === 0).map(([name]) => name);
		const entityDrops = unused(entityHits);
		const relationDrops = unused(relationHits);
		if (!entityDrops.length && !relationDrops.length) break;
		if (
			labels.entityLabels.length - entityDrops.length < 2 ||
			labels.relationLabels.length - relationDrops.length < 1
		) {
			removals.push({
				document: title,
				name: [...entityDrops, ...relationDrops].join(', '),
				kind: 'relation',
				reason: `kept despite 0 hits in round ${round}: dropping would leave too few labels`
			});
			break;
		}
		for (const name of entityDrops)
			removals.push({ document: title, name, kind: 'entity', reason: `0 hits in round ${round}` });
		for (const name of relationDrops)
			removals.push({
				document: title,
				name,
				kind: 'relation',
				reason: `0 hits in round ${round}`
			});
		labels.entityLabels = labels.entityLabels.filter((name) => !entityDrops.includes(name));
		labels.relationLabels = labels.relationLabels.filter((name) => !relationDrops.includes(name));
	}
	return labels;
}

const dictionary = JSON.parse(readFileSync(values.dictionary, 'utf8')) as {
	entries: DictionaryEntry[];
};
const entries = dictionary.entries.filter((entry) => entry.entityTypes && entry.relationTypes);
console.log(`${entries.length} dictionary entries from ${values.dictionary}.`);

const model = await activeChatModel();
const { value: grouping, attempts } = await generateValidated(
	model,
	buildPrompt(entries),
	RESPONSE_SCHEMA,
	MAX_TOKENS,
	parseGrouping
);
await disposeRuntime();
for (const [index, attempt] of attempts.entries()) {
	console.log(
		`Grouping attempt ${index + 1}: ${attempt.error ? `failed: ${attempt.error}` : 'valid'} after ${attempt.seconds.toFixed(1)}s.`
	);
}
if (!grouping) throw new Error('The model did not return a usable grouping in two attempts.');

const longestFirst = (names: string[]) => [...new Set(names)].sort((a, b) => b.length - a.length);
const unmatched: string[] = [];
const relationRename = renames(
	grouping.relationGroups,
	longestFirst(entries.flatMap((entry) => (entry.relationTypes ?? []).map((type) => type.name))),
	unmatched
);
console.log(`Relation renames: ${JSON.stringify(Object.fromEntries(relationRename))}`);

const removals: Removal[] = [];
const perDocument: Record<string, LabelSet> = {};
for (const entry of entries) {
	const start = documentLabels(entry, relationRename, removals);
	const rows = await db
		.select({ content: documentChunks.content })
		.from(documentChunks)
		.where(inArray(documentChunks.id, entry.sampleChunkIds));
	perDocument[entry.title] = await hitTest(
		entry.title,
		rows.map((row) => row.content),
		start,
		removals
	);
	console.log(
		`${entry.title}: ${perDocument[entry.title].entityLabels.length} entity, ` +
			`${perDocument[entry.title].relationLabels.length} relation labels`
	);
}

writeFileSync(values.out, JSON.stringify(perDocument, null, 2) + '\n');
writeFileSync(
	values.out.replace(/\.json$/, '-detail.json'),
	JSON.stringify(
		{
			dictionary: values.dictionary,
			grouping,
			unmatched,
			relationRenames: Object.fromEntries(relationRename),
			removals,
			perDocument,
			attempts
		},
		null,
		2
	) + '\n'
);
console.log(JSON.stringify({ removals, unmatched }, null, 2));
console.log(`Done: ${values.out}`);
process.exit(0);
