import { PipelineEmbeddingModel } from './pipeline-embedding-model';

// Compact multilingual model (25M active parameters). Trained without prefixes.
// "fp32" selects onnx/model.onnx, the model card's default (int8 token table, fp32 compute).
export const bekkoEmbeddingV1A25m = new PipelineEmbeddingModel({
	id: 'bekko-embedding-v1-a25m',
	name: 'bekko-embedding v1 a25m',
	repository: 'hotchpotch/bekko-embedding-v1-a25m',
	dtype: 'fp32',
	dimensions: 384,
	maxTokens: 8192,
	pooling: 'mean',
	prefixes: { search_document: '', search_query: '' }
});
