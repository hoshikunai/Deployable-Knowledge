// Chunk sizing is read once from the environment so benchmark corpora can be prepared with
// different chunk sizes. Unset, ingestion keeps the default character-based chunking.

import { RAG_CHUNK_CHARACTER_LIMIT } from '$lib/constants';

export type ChunkingConfig =
	| { unit: 'characters'; maxCharacters: number; overlapSentences: number }
	| { unit: 'tokens'; maxTokens: number; overlapTokens: number };

const DEFAULT_CHUNKING: ChunkingConfig = {
	unit: 'characters',
	maxCharacters: RAG_CHUNK_CHARACTER_LIMIT,
	overlapSentences: 1
};

const MIN_CHUNK_TOKENS = 16;
const MAX_CHUNK_TOKENS = 4096;

function readInteger(name: string): number | null {
	const raw = process.env[name]?.trim();
	if (!raw) return null;

	const value = Number(raw);
	if (!Number.isInteger(value) || value < 0) {
		throw new Error(`${name} must be a non-negative integer, received "${raw}".`);
	}

	return value;
}

function readChunkingConfig(): ChunkingConfig {
	const maxTokens = readInteger('RAG_CHUNK_MAX_TOKENS');
	const overlapTokens = readInteger('RAG_CHUNK_OVERLAP_TOKENS');

	if (maxTokens === null) {
		if (overlapTokens !== null) {
			throw new Error('RAG_CHUNK_OVERLAP_TOKENS requires RAG_CHUNK_MAX_TOKENS.');
		}
		return DEFAULT_CHUNKING;
	}

	if (maxTokens < MIN_CHUNK_TOKENS || maxTokens > MAX_CHUNK_TOKENS) {
		throw new Error(
			`RAG_CHUNK_MAX_TOKENS must be between ${MIN_CHUNK_TOKENS} and ${MAX_CHUNK_TOKENS}.`
		);
	}

	const overlap = overlapTokens ?? 0;
	if (overlap >= maxTokens) {
		throw new Error('RAG_CHUNK_OVERLAP_TOKENS must be smaller than RAG_CHUNK_MAX_TOKENS.');
	}

	return { unit: 'tokens', maxTokens, overlapTokens: overlap };
}

export const chunkingConfig: Readonly<ChunkingConfig> = Object.freeze(readChunkingConfig());
