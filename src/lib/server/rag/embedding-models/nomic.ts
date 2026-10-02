import { PipelineEmbeddingModel } from './pipeline-embedding-model';

// The app's default embedding model. It expects a task prefix on every input.
export const nomicEmbedTextV15 = new PipelineEmbeddingModel({
	id: 'nomic-embed-text-v1.5',
	name: 'nomic-embed-text v1.5',
	repository: 'nomic-ai/nomic-embed-text-v1.5',
	dtype: 'q8',
	dimensions: 768,
	maxTokens: 8192,
	pooling: 'mean',
	prefixes: { search_document: 'search_document: ', search_query: 'search_query: ' }
});
