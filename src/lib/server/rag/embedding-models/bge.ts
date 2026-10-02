import { PipelineEmbeddingModel } from './pipeline-embedding-model';

// BGE v1.5 English models. Queries get the retrieval instruction from the model card;
// passages are embedded as-is. Xenova's repositories are the Transformers.js conversions.
// Full precision matches the official vectors; the 8-bit files reorder near-tied passages.
const BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';

export const bgeBaseEnV15 = new PipelineEmbeddingModel({
	id: 'bge-base-en-v1.5',
	name: 'bge-base-en v1.5',
	repository: 'Xenova/bge-base-en-v1.5',
	dtype: 'fp32',
	dimensions: 768,
	maxTokens: 512,
	pooling: 'cls',
	prefixes: { search_document: '', search_query: BGE_QUERY_INSTRUCTION }
});

export const bgeSmallEnV15 = new PipelineEmbeddingModel({
	id: 'bge-small-en-v1.5',
	name: 'bge-small-en v1.5',
	repository: 'Xenova/bge-small-en-v1.5',
	dtype: 'fp32',
	dimensions: 384,
	maxTokens: 512,
	pooling: 'cls',
	prefixes: { search_document: '', search_query: BGE_QUERY_INSTRUCTION }
});
