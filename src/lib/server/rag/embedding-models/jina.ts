import { PipelineEmbeddingModel } from './pipeline-embedding-model';

// Jina embeddings v5 text nano, retrieval adapter merged (EuroBERT). Licensed CC BY-NC 4.0:
// for benchmarking only; it cannot become the app default without a commercial Jina license.
export const jinaEmbeddingsV5TextNanoRetrieval = new PipelineEmbeddingModel({
	id: 'jina-embeddings-v5-text-nano-retrieval',
	name: 'jina-embeddings v5 text nano retrieval',
	repository: 'jinaai/jina-embeddings-v5-text-nano-retrieval',
	dtype: 'q8',
	dimensions: 768,
	maxTokens: 8192,
	pooling: 'last_token',
	prefixes: { search_document: 'Document: ', search_query: 'Query: ' }
});
