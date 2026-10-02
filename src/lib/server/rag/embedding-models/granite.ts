import { PipelineEmbeddingModel } from './pipeline-embedding-model';

// IBM Granite multilingual R2 (97M, ModernBERT). Trained without prefixes. Full precision:
// the repository's 8-bit file drifts too far from the official vectors (cosine ~0.95).
export const graniteEmbedding97mMultilingualR2 = new PipelineEmbeddingModel({
	id: 'granite-embedding-97m-multilingual-r2',
	name: 'granite-embedding 97m multilingual r2',
	repository: 'ibm-granite/granite-embedding-97m-multilingual-r2',
	dtype: 'fp32',
	dimensions: 384,
	maxTokens: 32768,
	pooling: 'cls',
	prefixes: { search_document: '', search_query: '' }
});

// IBM Granite English R2 (149M, ModernBERT). IBM publishes no ONNX export, so this loads the
// onnx-community conversion. Full precision: its 8-bit file drifts too far (cosine ~0.975).
export const graniteEmbeddingEnglishR2 = new PipelineEmbeddingModel({
	id: 'granite-embedding-english-r2',
	name: 'granite-embedding english r2',
	repository: 'onnx-community/granite-embedding-english-r2-ONNX',
	dtype: 'fp32',
	dimensions: 768,
	maxTokens: 8192,
	pooling: 'cls',
	prefixes: { search_document: '', search_query: '' }
});
