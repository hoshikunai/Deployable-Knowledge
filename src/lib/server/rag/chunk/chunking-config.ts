import { RAG_CHUNK_CHARACTER_LIMIT } from '#lib/constants';
import { getActiveEmbedding } from '../embedding-model';
import { characterBudget, tokenBudget, type ChunkBudget } from './chunker';

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

export async function loadChunkBudget(): Promise<ChunkBudget> {
	if (chunkingConfig.unit === 'characters') {
		return characterBudget(chunkingConfig.maxCharacters, chunkingConfig.overlapSentences);
	}

	const { provider, settings } = await getActiveEmbedding();
	const counter = await provider.createTokenCounter(settings.model, settings.device);

	if (chunkingConfig.maxTokens > counter.maxTokens) {
		throw new Error(
			`RAG_CHUNK_MAX_TOKENS ${chunkingConfig.maxTokens} exceeds ${settings.model}'s ` +
				`limit of ${counter.maxTokens} tokens.`
		);
	}

	return tokenBudget(chunkingConfig.maxTokens, chunkingConfig.overlapTokens, counter.count);
}
