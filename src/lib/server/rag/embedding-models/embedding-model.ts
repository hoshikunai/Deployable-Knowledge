import type { ProgressCallback } from '@huggingface/transformers';

// Queries and stored chunks can be formatted differently (e.g. nomic's prefixes)
export type EmbeddingPurpose = 'search_document' | 'search_query';

export type EmbeddingProgress = (current: number, total: number) => void;

// Any dense embedding model the RAG pipeline can use. Vectors must be L2-normalized,
// because semantic search scores them with a plain dot product.
export interface EmbeddingModel {
	// Stable ID, recorded by benchmarks and selected with RAG_EMBEDDING_MODEL
	readonly id: string;
	readonly name: string;
	// Hugging Face repository and weight variant, shown by first-run setup
	readonly repository: string;
	readonly dtype: string;
	readonly dimensions: number;
	// Longest input the model reads; token-sized chunks must fit within it
	readonly maxTokens: number;

	isInstalled(): Promise<boolean>;
	install(onProgress: ProgressCallback): Promise<void>;
	embed(
		texts: readonly string[],
		purpose: EmbeddingPurpose,
		onProgress?: EmbeddingProgress
	): Promise<Float32Array[]>;
	// Counts tokens as the model sees them, without special tokens; used to size chunks
	createTokenCounter(): Promise<(text: string) => number>;
}
