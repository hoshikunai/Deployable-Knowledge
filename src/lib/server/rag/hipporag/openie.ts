import type { Provider } from '$lib/server/providers/provider';
import type { HippoOpenIeResult, HippoTriple } from './types';

const OPENIE_MAX_TOKENS = 2_048;

function compactEntity(value: string): string {
	return value.replace(/\s+/g, ' ').trim();
}

export function normalizeEntity(value: string): string {
	return compactEntity(value).toLocaleLowerCase();
}

function parseJsonObject(output: string): Record<string, unknown> {
	const fenced = output.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
	const source = fenced ?? output.slice(output.indexOf('{'), output.lastIndexOf('}') + 1);
	const parsed: unknown = JSON.parse(source);

	if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
		throw new Error('The extraction model did not return a JSON object.');
	}

	return parsed as Record<string, unknown>;
}

async function complete(provider: Provider, modelId: string, prompt: string, signal?: AbortSignal) {
	let output = '';
	for await (const text of provider.chat(prompt, modelId, {
		maxTokens: OPENIE_MAX_TOKENS,
		reasoningBudget: 0,
		temperature: 0,
		signal
	})) {
		output += text;
	}
	return output;
}

function readEntities(value: unknown): string[] {
	if (!Array.isArray(value)) throw new Error('OpenIE entities must be an array.');

	const entities = new Map<string, string>();
	for (const item of value) {
		if (typeof item !== 'string') throw new Error('Every OpenIE entity must be a string.');
		const name = compactEntity(item);
		if (name) entities.set(normalizeEntity(name), name);
	}
	return [...entities.values()];
}

function readTriples(value: unknown): HippoTriple[] {
	if (!Array.isArray(value)) throw new Error('OpenIE triples must be an array.');

	const triples: HippoTriple[] = [];
	for (const item of value) {
		if (!item || typeof item !== 'object' || Array.isArray(item)) {
			throw new Error('Every OpenIE triple must be an object.');
		}
		const row = item as Record<string, unknown>;
		if (
			typeof row.subject !== 'string' ||
			typeof row.predicate !== 'string' ||
			typeof row.object !== 'string'
		) {
			throw new Error('OpenIE triples require string subject, predicate, and object fields.');
		}

		const triple = {
			subject: compactEntity(row.subject),
			predicate: compactEntity(row.predicate),
			object: compactEntity(row.object)
		};
		if (triple.subject && triple.predicate && triple.object) triples.push(triple);
	}
	return triples;
}

export async function extractOpenIe(
	provider: Provider,
	modelId: string,
	passage: string,
	signal?: AbortSignal
): Promise<HippoOpenIeResult> {
	const entityOutput = await complete(
		provider,
		modelId,
		`Extract the named entities and important concepts from the passage. Preserve the wording used in the passage. Return only valid JSON with this shape: {"entities":["entity"]}.\n\nPASSAGE:\n${passage}`,
		signal
	);
	const entities = readEntities(parseJsonObject(entityOutput).entities);
	const tripleOutput = await complete(
		provider,
		modelId,
		`Extract atomic factual relations from the passage as subject-predicate-object triples. Resolve pronouns to explicit entities when the passage supports it. Use the supplied entity names where possible. Return only valid JSON with this shape: {"triples":[{"subject":"...","predicate":"...","object":"..."}]}. Do not add facts that are not stated.\n\nKNOWN ENTITIES:\n${entities.join(', ')}\n\nPASSAGE:\n${passage}`,
		signal
	);
	const triples = readTriples(parseJsonObject(tripleOutput).triples);
	const byNormalizedName = new Map(entities.map((entity) => [normalizeEntity(entity), entity]));

	for (const triple of triples) {
		byNormalizedName.set(normalizeEntity(triple.subject), triple.subject);
		byNormalizedName.set(normalizeEntity(triple.object), triple.object);
	}

	return { entities: [...byNormalizedName.values()], triples };
}

export async function recognizeFacts(
	provider: Provider,
	modelId: string,
	query: string,
	facts: readonly string[],
	signal?: AbortSignal
): Promise<number[]> {
	if (facts.length === 0) return [];

	const numberedFacts = facts.map((fact, index) => `${index}: ${fact}`).join('\n');
	const output = await complete(
		provider,
		modelId,
		`Select facts that mention an entity or concept recognized in the query. Return only valid JSON with this shape: {"indices":[0]}. Use zero-based indices from the list. Return an empty array when none apply.\n\nQUERY:\n${query}\n\nFACTS:\n${numberedFacts}`,
		signal
	);
	const value = parseJsonObject(output).indices;
	if (!Array.isArray(value)) throw new Error('Recognition indices must be an array.');

	return [
		...new Set(
			value.filter(
				(index): index is number =>
					Number.isInteger(index) && Number(index) >= 0 && Number(index) < facts.length
			)
		)
	];
}
