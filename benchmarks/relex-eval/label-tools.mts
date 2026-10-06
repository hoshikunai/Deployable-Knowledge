import { eq } from 'drizzle-orm';
import { db } from '../../src/lib/server/database/database';
import { appState, profiles } from '../../src/lib/server/database/schema';
import { extractRelex } from '../../src/lib/server/entities/relex';
import type { ProviderJsonSchema } from '../../src/lib/server/providers/provider';
import { findProvider } from '../../src/lib/server/providers/registry';

export interface RelationExample {
	head: string;
	tail: string;
	sentence: string;
}

export interface ProposedType {
	name: string;
	definition: string;
	head?: string;
	tail?: string;
	example?: RelationExample;
	grounded?: boolean;
}

export interface LabelSet {
	entityLabels: string[];
	relationLabels: string[];
}

export interface ReadContext {
	excerptText: string;
	rejected: string[];
	maxLabels: number;
}

export interface GenerationAttempt {
	seconds: number;
	output: string;
	error?: string;
}

export const GENERIC_NAMES = new Set([
	'object',
	'thing',
	'item',
	'entity',
	'concept',
	'other',
	'information',
	'data',
	'text',
	'term'
]);

export const VAGUE_RELATION_PREFIXES = [
	'is located',
	'located',
	'contains',
	'includes',
	'has',
	'is part of',
	'part of',
	'applies to',
	'is related to',
	'related to',
	'is associated with',
	'associated with',
	'is classified as',
	'dated',
	'is dated',
	'involves',
	'is detailed in',
	'is described in',
	'is defined in',
	'is listed in',
	'is shown in',
	'is referenced in',
	'is mentioned in',
	'is contained in'
];

export const entityTypeSchema = {
	type: 'object',
	properties: { name: { type: 'string' }, definition: { type: 'string' } },
	required: ['name', 'definition']
};

export const relationTypeSchema = {
	type: 'object',
	properties: {
		name: { type: 'string' },
		definition: { type: 'string' },
		head: { type: 'string' },
		tail: { type: 'string' },
		example: {
			type: 'object',
			properties: {
				head: { type: 'string' },
				tail: { type: 'string' },
				sentence: { type: 'string' }
			},
			required: ['head', 'tail', 'sentence']
		}
	},
	required: ['name', 'definition', 'head', 'tail', 'example']
};

export function seededRandom(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let value = state;
		value = Math.imul(value ^ (value >>> 15), value | 1);
		value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
		return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
	};
}

function dot(matrix: Float32Array, row: number, centre: Float32Array, dimensions: number): number {
	let total = 0;
	for (let index = 0; index < dimensions; index += 1)
		total += matrix[row * dimensions + index] * centre[index];
	return total;
}

export function representativeRows(
	matrix: Float32Array,
	rows: number[],
	dimensions: number,
	clusters: number,
	random: () => number
): number[] {
	const centres = Array.from({ length: clusters }, () => {
		const row = rows[Math.floor(random() * rows.length)];
		return matrix.slice(row * dimensions, (row + 1) * dimensions);
	});
	const assignment = new Int32Array(rows.length);
	for (let iteration = 0; iteration < 12; iteration += 1) {
		rows.forEach((row, position) => {
			let best = 0;
			let bestScore = -Infinity;
			centres.forEach((centre, cluster) => {
				const score = dot(matrix, row, centre, dimensions);
				if (score > bestScore) {
					bestScore = score;
					best = cluster;
				}
			});
			assignment[position] = best;
		});
		centres.forEach((centre, cluster) => {
			const sums = new Float32Array(dimensions);
			let members = 0;
			rows.forEach((row, position) => {
				if (assignment[position] !== cluster) return;
				members += 1;
				for (let index = 0; index < dimensions; index += 1)
					sums[index] += matrix[row * dimensions + index];
			});
			if (members === 0) return;
			const norm = Math.hypot(...sums) || 1;
			for (let index = 0; index < dimensions; index += 1) centre[index] = sums[index] / norm;
		});
	}
	return centres
		.map((centre, cluster) => {
			let best = -1;
			let bestScore = -Infinity;
			rows.forEach((row, position) => {
				if (assignment[position] !== cluster) return;
				const score = dot(matrix, row, centre, dimensions);
				if (score > bestScore) {
					bestScore = score;
					best = row;
				}
			});
			return best;
		})
		.filter((row) => row >= 0);
}

export function normalizeText(text: string): string {
	return text.replace(/\s+/g, ' ').trim().toLowerCase();
}

export function normalizeLabel(name: string): string {
	return name
		.trim()
		.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.toLowerCase();
}

export function isVagueRelation(name: string): boolean {
	return VAGUE_RELATION_PREFIXES.some((prefix) => name === prefix || name.startsWith(`${prefix} `));
}

function readExample(value: unknown): RelationExample | undefined {
	if (!value || typeof value !== 'object') return undefined;
	const record = value as Record<string, unknown>;
	if (
		typeof record.head !== 'string' ||
		typeof record.tail !== 'string' ||
		typeof record.sentence !== 'string'
	) {
		return undefined;
	}
	return { head: record.head, tail: record.tail, sentence: record.sentence };
}

export function readTypes(value: unknown, kind: string, context: ReadContext): ProposedType[] {
	if (!Array.isArray(value)) throw new Error(`'${kind}' is not a list.`);
	const seen = new Set<string>();
	const types: ProposedType[] = [];
	for (const item of value) {
		if (!item || typeof item !== 'object')
			throw new Error(`An entry in '${kind}' is not an object.`);
		const record = item as Record<string, unknown>;
		if (typeof record.name !== 'string') throw new Error(`An entry in '${kind}' has no name.`);
		const name = normalizeLabel(record.name);
		if (!name || name.length > 40 || name.split(/\s+/).length > 5) continue;
		if (GENERIC_NAMES.has(name) || seen.has(name)) continue;
		if (kind === 'relationTypes' && isVagueRelation(name)) {
			context.rejected.push(name);
			continue;
		}
		seen.add(name);
		const example = readExample(record.example);
		types.push({
			name,
			definition: typeof record.definition === 'string' ? record.definition : '',
			...(typeof record.head === 'string' ? { head: record.head } : {}),
			...(typeof record.tail === 'string' ? { tail: record.tail } : {}),
			...(example
				? {
						example,
						grounded: context.excerptText.includes(normalizeText(example.sentence))
					}
				: {})
		});
	}
	return types.slice(0, context.maxLabels);
}

export function parseJsonObject(output: string): Record<string, unknown> {
	const start = output.indexOf('{');
	const end = output.lastIndexOf('}');
	if (start < 0 || end <= start) throw new Error('The reply contains no JSON object.');
	return JSON.parse(output.slice(start, end + 1)) as Record<string, unknown>;
}

export async function activeChatModel() {
	const [state] = await db.select().from(appState).where(eq(appState.id, 'app'));
	const [profile] = await db.select().from(profiles).where(eq(profiles.id, state.activeProfileId!));
	const provider = await findProvider(profile.provider);
	if (!provider) throw new Error(`Provider '${profile.provider}' is not available.`);
	return { profile, provider };
}

export async function generateValidated<T>(
	model: Awaited<ReturnType<typeof activeChatModel>>,
	prompt: string,
	schema: ProviderJsonSchema,
	maxTokens: number,
	parse: (output: string) => T
): Promise<{ value?: T; attempts: GenerationAttempt[] }> {
	const attempts: GenerationAttempt[] = [];
	for (let attempt = 1; attempt <= 2; attempt += 1) {
		const started = performance.now();
		let output = '';
		for await (const part of model.provider.chat(prompt, model.profile.model, {
			temperature: model.profile.temperature,
			topK: model.profile.topK,
			maxTokens,
			reasoningBudget: 0,
			gpuMode: model.profile.gpuMode,
			responseSchema: schema
		})) {
			output += part;
		}
		const record: GenerationAttempt = { seconds: (performance.now() - started) / 1000, output };
		attempts.push(record);
		try {
			return { value: parse(output), attempts };
		} catch (error) {
			record.error = error instanceof Error ? error.message : String(error);
		}
	}
	return { attempts };
}

export async function countHits(texts: string[], labels: LabelSet) {
	const entityHits = new Map(labels.entityLabels.map((name) => [name, 0]));
	const relationHits = new Map(labels.relationLabels.map((name) => [name, 0]));
	let relations = 0;
	for (const text of texts) {
		const result = await extractRelex(text, labels);
		relations += result.relations.length;
		for (const label of new Set(result.entities.map((entity) => entity.label))) {
			entityHits.set(label, (entityHits.get(label) ?? 0) + 1);
		}
		for (const label of new Set(result.relations.map((relation) => relation.relation))) {
			relationHits.set(label, (relationHits.get(label) ?? 0) + 1);
		}
	}
	return { entityHits, relationHits, relations };
}
