import { setImmediate as yieldEventLoop } from 'node:timers/promises';
import {
	ModelRegistry,
	pipeline,
	type DataType,
	type FeatureExtractionPipeline
} from '@huggingface/transformers';
import type { EmbeddingTask } from '#lib/constants';
import { createLogger } from '#lib/server/logging';
import { INFERENCE_THREADS } from '#lib/server/utils/inference-threads';
import { TRANSFORMERS_CACHE_DIR } from '#lib/server/utils/transformers-env';
import { EmbeddingProvider, type TokenCounter } from './provider';

export const TRANSFORMERS_EMBEDDING_PROVIDER_ID = 'transformers';

const BATCH_SIZE = 16;
const BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';

type BenchmarkEmbeddingModel = {
	id: string;
	repository: string;
	dtype: DataType;
	maxTokens: number;
	prefixes: Record<EmbeddingTask, string>;
};

export const BENCHMARK_EMBEDDING_MODELS: readonly BenchmarkEmbeddingModel[] = [
	{
		id: 'bge-small-en-v1.5',
		repository: 'Xenova/bge-small-en-v1.5',
		dtype: 'fp32',
		maxTokens: 512,
		prefixes: { search_document: '', search_query: BGE_QUERY_INSTRUCTION }
	}
];

const log = createLogger('embedding');

function catalogModel(id: string): BenchmarkEmbeddingModel {
	const model = BENCHMARK_EMBEDDING_MODELS.find((candidate) => candidate.id === id);
	if (!model) {
		const known = BENCHMARK_EMBEDDING_MODELS.map((candidate) => candidate.id).join(', ');
		throw new Error(`Embedding model "${id}" is not a benchmark model (known: ${known}).`);
	}
	return model;
}

export class TransformersEmbeddingProvider extends EmbeddingProvider {
	override id = TRANSFORMERS_EMBEDDING_PROVIDER_ID;
	override name = 'Transformers.js (benchmark only)';

	private extractors = new Map<string, Promise<FeatureExtractionPipeline>>();

	override isInstalled(id: string): Promise<boolean> {
		const model = catalogModel(id);
		return ModelRegistry.is_pipeline_cached('feature-extraction', model.repository, {
			cache_dir: TRANSFORMERS_CACHE_DIR,
			dtype: model.dtype
		});
	}

	override async createTokenCounter(id: string): Promise<TokenCounter> {
		const { tokenizer } = await this.getExtractor(id);
		return {
			maxTokens: catalogModel(id).maxTokens,
			count: (text) => tokenizer.encode(text, { add_special_tokens: false }).length
		};
	}

	override async embed(texts: string[], task: EmbeddingTask, id: string): Promise<Float32Array[]> {
		if (texts.length === 0) return [];

		const model = catalogModel(id);
		const extractor = await this.getExtractor(id);
		const vectors: Float32Array[] = [];

		for (let index = 0; index < texts.length; index += BATCH_SIZE) {
			const batch = texts
				.slice(index, index + BATCH_SIZE)
				.map((text) => model.prefixes[task] + text);

			const output = await extractor(batch, { pooling: 'cls', normalize: true });
			const values = output.data as Float32Array;
			const [rows, dims] = output.dims as [number, number];
			for (let row = 0; row < rows; row += 1) {
				vectors.push(Float32Array.from(values.subarray(row * dims, (row + 1) * dims)));
			}
			output.dispose();

			await yieldEventLoop();
		}

		return vectors;
	}

	private getExtractor(id: string): Promise<FeatureExtractionPipeline> {
		const existing = this.extractors.get(id);
		if (existing) return existing;

		const model = catalogModel(id);
		const started = Date.now();
		const loading = pipeline('feature-extraction', model.repository, {
			dtype: model.dtype,
			cache_dir: TRANSFORMERS_CACHE_DIR,
			session_options: { intraOpNumThreads: INFERENCE_THREADS, interOpNumThreads: 1 }
		})
			.then((extractor) => {
				log.info('Embedding model ready', { durationMs: Date.now() - started, model: id });
				return extractor;
			})
			.catch((error: unknown) => {
				this.extractors.delete(id);
				log.error('Embedding model failed to load', { model: id }, error);
				throw error;
			});

		this.extractors.set(id, loading);
		return loading;
	}
}
