import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import type { GoldAssertion, GoldBenchmark, GoldChunk, GoldQuery } from './benchmark-types';

export interface CorpusChunk {
	chunkId: string;
	documentId: string;
	content: string;
}

export async function loadGoldBenchmark(path: string): Promise<GoldBenchmark> {
	const value: unknown = JSON.parse(await readFile(path, 'utf8'));
	assertGoldBenchmark(value);
	return value;
}

export function validateGoldBenchmark(benchmark: GoldBenchmark): string[] {
	const errors: string[] = [];
	const chunkIds = new Set<string>();
	const assertionIds = new Set<string>();
	const queryIds = new Set<string>();
	const relations = new Set(benchmark.canonicalRelations);

	if (!benchmark.version.trim()) errors.push('version must not be empty.');
	if (!benchmark.corpusId.trim()) errors.push('corpusId must not be empty.');
	if (!benchmark.canonicalRelations.length) {
		errors.push('canonicalRelations must contain at least one relation.');
	}

	for (const relation of benchmark.canonicalRelations) {
		if (!relation.trim()) errors.push('canonicalRelations cannot contain an empty relation.');
	}
	if (relations.size !== benchmark.canonicalRelations.length) {
		errors.push('canonicalRelations contains duplicates.');
	}

	for (const chunk of benchmark.chunks) {
		if (chunkIds.has(chunk.chunkId)) errors.push(`Duplicate chunk ID: ${chunk.chunkId}`);
		chunkIds.add(chunk.chunkId);
		if (!/^[a-f\d]{64}$/i.test(chunk.contentSha256)) {
			errors.push(`Chunk ${chunk.chunkId} does not have a 64-character SHA-256 hash.`);
		}
		if (hasPlaceholder(chunk.chunkId) || hasPlaceholder(chunk.documentId)) {
			errors.push(`Chunk ${chunk.chunkId} still contains a placeholder ID.`);
		}
	}

	for (const assertion of benchmark.assertions) {
		if (assertionIds.has(assertion.id)) errors.push(`Duplicate assertion ID: ${assertion.id}`);
		assertionIds.add(assertion.id);
		if (!chunkIds.has(assertion.chunkId)) {
			errors.push(`Assertion ${assertion.id} references unknown chunk ${assertion.chunkId}.`);
		}
		const chunk = benchmark.chunks.find((candidate) => candidate.chunkId === assertion.chunkId);
		if (chunk && chunk.documentId !== assertion.documentId) {
			errors.push(`Assertion ${assertion.id} has a document ID that differs from its chunk.`);
		}
		if (!relations.has(assertion.predicate)) {
			errors.push(`Assertion ${assertion.id} uses non-canonical relation ${assertion.predicate}.`);
		}
		if (!assertion.subject.canonical.trim() || !assertion.object.canonical.trim()) {
			errors.push(`Assertion ${assertion.id} has an empty endpoint.`);
		}
		if (!assertion.evidence.trim()) errors.push(`Assertion ${assertion.id} has empty evidence.`);
	}

	for (const query of benchmark.queries) {
		if (queryIds.has(query.id)) errors.push(`Duplicate query ID: ${query.id}`);
		queryIds.add(query.id);
		if (!query.question.trim()) errors.push(`Query ${query.id} has an empty question.`);
		if (!query.relevantChunkIds.length) {
			errors.push(`Query ${query.id} has no relevant chunks.`);
		}
		for (const chunkId of [...query.relevantChunkIds, ...(query.forbiddenChunkIds ?? [])]) {
			if (!chunkIds.has(chunkId))
				errors.push(`Query ${query.id} references unknown chunk ${chunkId}.`);
		}
		for (const path of query.expectedPathAssertionIds) {
			if (!path.length) errors.push(`Query ${query.id} contains an empty expected path.`);
			for (const assertionId of path) {
				if (!assertionIds.has(assertionId)) {
					errors.push(`Query ${query.id} references unknown assertion ${assertionId}.`);
				}
			}
		}
	}

	return errors;
}

export function validateCorpusChunks(
	benchmark: GoldBenchmark,
	corpusChunks: CorpusChunk[]
): string[] {
	const errors: string[] = [];
	const actual = new Map(corpusChunks.map((chunk) => [chunk.chunkId, chunk]));

	for (const expected of benchmark.chunks) {
		const chunk = actual.get(expected.chunkId);
		if (!chunk) {
			errors.push(`Benchmark chunk ${expected.chunkId} is missing from the corpus.`);
			continue;
		}
		if (chunk.documentId !== expected.documentId) {
			errors.push(`Benchmark chunk ${expected.chunkId} belongs to a different document.`);
		}
		const digest = createHash('sha256').update(chunk.content).digest('hex');
		if (digest !== expected.contentSha256.toLowerCase()) {
			errors.push(`Benchmark chunk ${expected.chunkId} has changed (SHA-256 mismatch).`);
		}
		for (const assertion of benchmark.assertions.filter(
			(candidate) => candidate.chunkId === expected.chunkId
		)) {
			if (!chunk.content.includes(assertion.evidence)) {
				errors.push(`Assertion ${assertion.id} evidence is not verbatim in its source chunk.`);
			}
		}
	}

	return errors;
}

function assertGoldBenchmark(value: unknown): asserts value is GoldBenchmark {
	if (!isRecord(value)) throw new Error('The benchmark root must be a JSON object.');
	if (typeof value.version !== 'string' || typeof value.corpusId !== 'string') {
		throw new Error('The benchmark requires string version and corpusId fields.');
	}
	if (!isStringArray(value.canonicalRelations)) {
		throw new Error('canonicalRelations must be an array of strings.');
	}
	if (!Array.isArray(value.chunks) || !value.chunks.every(isGoldChunk)) {
		throw new Error('chunks contains an invalid benchmark chunk.');
	}
	if (!Array.isArray(value.assertions) || !value.assertions.every(isGoldAssertion)) {
		throw new Error('assertions contains an invalid gold assertion.');
	}
	if (!Array.isArray(value.queries) || !value.queries.every(isGoldQuery)) {
		throw new Error('queries contains an invalid gold query.');
	}
}

function isGoldChunk(value: unknown): value is GoldChunk {
	return (
		isRecord(value) &&
		typeof value.chunkId === 'string' &&
		typeof value.documentId === 'string' &&
		typeof value.contentSha256 === 'string'
	);
}

function isGoldAssertion(value: unknown): value is GoldAssertion {
	return (
		isRecord(value) &&
		typeof value.id === 'string' &&
		typeof value.documentId === 'string' &&
		typeof value.chunkId === 'string' &&
		isGoldEntity(value.subject) &&
		typeof value.predicate === 'string' &&
		isGoldEntity(value.object) &&
		typeof value.evidence === 'string' &&
		(value.status === 'asserted' || value.status === 'negated' || value.status === 'uncertain') &&
		isAssertionModality(value.modality) &&
		typeof value.required === 'boolean'
	);
}

function isAssertionModality(value: unknown): boolean {
	return (
		value === 'observed' ||
		value === 'habitual' ||
		value === 'required' ||
		value === 'recommended' ||
		value === 'permitted' ||
		value === 'prohibited'
	);
}

function isGoldEntity(value: unknown): boolean {
	return (
		isRecord(value) &&
		typeof value.canonical === 'string' &&
		typeof value.type === 'string' &&
		(value.aliases === undefined || isStringArray(value.aliases))
	);
}

function isGoldQuery(value: unknown): value is GoldQuery {
	return (
		isRecord(value) &&
		typeof value.id === 'string' &&
		typeof value.question === 'string' &&
		isStringArray(value.relevantChunkIds) &&
		Array.isArray(value.expectedPathAssertionIds) &&
		value.expectedPathAssertionIds.every(isStringArray) &&
		(value.forbiddenChunkIds === undefined || isStringArray(value.forbiddenChunkIds))
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function hasPlaceholder(value: string): boolean {
	return value.toLowerCase().includes('replace-with');
}
