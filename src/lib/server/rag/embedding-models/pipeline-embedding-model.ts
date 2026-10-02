// Embedding models that the standard Transformers.js feature-extraction pipeline can run.
// Models needing custom pooling or inputs can implement EmbeddingModel directly instead.

import { setImmediate as yieldEventLoop } from 'node:timers/promises';
import {
	ModelRegistry,
	pipeline,
	type DataType,
	type FeatureExtractionPipeline,
	type ProgressCallback
} from '@huggingface/transformers';
import { diagnosticEvents } from '$lib/server/diagnostics/events';
import { INFERENCE_THREADS, MODEL_CACHE_DIR } from '../inference';
import type { EmbeddingModel, EmbeddingProgress, EmbeddingPurpose } from './embedding-model';

const EMBEDDING_BATCH_SIZE = 16;

export type PipelineEmbeddingConfig = {
	id: string;
	name: string;
	repository: string;
	dtype: DataType;
	dimensions: number;
	maxTokens: number;
	pooling: 'mean' | 'cls' | 'last_token';
	// Text placed before each input, per purpose
	prefixes: Record<EmbeddingPurpose, string>;
};

// The pipeline's own last_token pooling reads the final padded position, so shorter texts in a
// batch would get a padding vector; take each row's last attended token instead.
async function embedLastTokens(
	extractor: FeatureExtractionPipeline,
	batch: string[]
): Promise<Float32Array[]> {
	const output = await extractor(batch, { pooling: 'none' });
	const { attention_mask: mask } = extractor.tokenizer(batch, { padding: true, truncation: true });
	const values = output.data as Float32Array;
	const [rows, length, dims] = output.dims as [number, number, number];
	const attended = mask.data as BigInt64Array;
	const vectors: Float32Array[] = [];

	for (let row = 0; row < rows; row += 1) {
		let position = length - 1;
		while (position > 0 && attended[row * length + position] === 0n) position -= 1;

		const offset = (row * length + position) * dims;
		const vector = Float32Array.from(values.subarray(offset, offset + dims));
		const norm = Math.hypot(...vector);
		vectors.push(vector.map((value) => value / norm));
	}

	output.dispose();
	mask.dispose();
	return vectors;
}

export class PipelineEmbeddingModel implements EmbeddingModel {
	readonly id: string;
	readonly name: string;
	readonly repository: string;
	readonly dtype: DataType;
	readonly dimensions: number;
	readonly maxTokens: number;

	private readonly config: PipelineEmbeddingConfig;
	private extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

	constructor(config: PipelineEmbeddingConfig) {
		this.config = config;
		this.id = config.id;
		this.name = config.name;
		this.repository = config.repository;
		this.dtype = config.dtype;
		this.dimensions = config.dimensions;
		this.maxTokens = config.maxTokens;
	}

	isInstalled(): Promise<boolean> {
		return ModelRegistry.is_pipeline_cached('feature-extraction', this.repository, {
			cache_dir: MODEL_CACHE_DIR,
			dtype: this.dtype
		});
	}

	async install(onProgress: ProgressCallback): Promise<void> {
		await this.getExtractor(onProgress);
	}

	async createTokenCounter(): Promise<(text: string) => number> {
		const { tokenizer } = await this.getExtractor();
		return (text) => tokenizer.encode(text, { add_special_tokens: false }).length;
	}

	async embed(
		texts: readonly string[],
		purpose: EmbeddingPurpose,
		onProgress?: EmbeddingProgress
	): Promise<Float32Array[]> {
		if (texts.length === 0) return [];

		const extractor = await this.getExtractor();
		const prefix = this.config.prefixes[purpose];
		const embeddings: Float32Array[] = [];

		for (let index = 0; index < texts.length; index += EMBEDDING_BATCH_SIZE) {
			const batch = texts.slice(index, index + EMBEDDING_BATCH_SIZE).map((text) => prefix + text);

			if (this.config.pooling === 'last_token') {
				embeddings.push(...(await embedLastTokens(extractor, batch)));
			} else {
				const output = await extractor(batch, { pooling: this.config.pooling, normalize: true });
				const values = output.data as Float32Array;
				const [rows, dims] = output.dims as [number, number];
				for (let row = 0; row < rows; row += 1) {
					embeddings.push(Float32Array.from(values.subarray(row * dims, (row + 1) * dims)));
				}
				output.dispose();
			}

			onProgress?.(Math.min(index + EMBEDDING_BATCH_SIZE, texts.length), texts.length);
			await yieldEventLoop();
		}

		return embeddings;
	}

	private getExtractor(onProgress?: ProgressCallback): Promise<FeatureExtractionPipeline> {
		if (!this.extractorPromise) {
			const started = Date.now();
			console.log(`[Embedding] Loading ${this.repository} on ${INFERENCE_THREADS} thread(s)...`);
			this.extractorPromise = pipeline('feature-extraction', this.repository, {
				dtype: this.dtype,
				cache_dir: MODEL_CACHE_DIR,
				session_options: { intraOpNumThreads: INFERENCE_THREADS, interOpNumThreads: 1 },
				progress_callback: onProgress
			})
				.then((loaded) => {
					console.log(`[Embedding] Model ready in ${((Date.now() - started) / 1000).toFixed(1)}s.`);
					diagnosticEvents.embeddingReady(Date.now() - started);
					return loaded;
				})
				.catch((error) => {
					console.error('[Embedding] Model failed to load.');
					diagnosticEvents.embeddingFailed();
					this.extractorPromise = undefined;
					throw error;
				});
		}

		return this.extractorPromise;
	}
}
