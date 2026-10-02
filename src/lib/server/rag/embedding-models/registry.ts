// The active embedding model is read once from RAG_EMBEDDING_MODEL so benchmarks can
// prepare corpora with different models. Unset, the app uses nomic-embed-text v1.5.
// To add a model: create its EmbeddingModel and list it here.

import type { EmbeddingModel } from './embedding-model';
import { bekkoEmbeddingV1A25m } from './bekko';
import { bgeBaseEnV15, bgeSmallEnV15 } from './bge';
import { graniteEmbedding97mMultilingualR2, graniteEmbeddingEnglishR2 } from './granite';
import { jinaEmbeddingsV5TextNanoRetrieval } from './jina';
import { nomicEmbedTextV15 } from './nomic';

const DEFAULT_EMBEDDING_MODEL = nomicEmbedTextV15;

const builtInEmbeddingModels: readonly EmbeddingModel[] = [
	nomicEmbedTextV15,
	bekkoEmbeddingV1A25m,
	bgeBaseEnV15,
	bgeSmallEnV15,
	graniteEmbedding97mMultilingualR2,
	graniteEmbeddingEnglishR2,
	jinaEmbeddingsV5TextNanoRetrieval
];

function readActiveEmbeddingModel(): EmbeddingModel {
	const configured = process.env.RAG_EMBEDDING_MODEL?.trim();
	if (!configured) return DEFAULT_EMBEDDING_MODEL;

	const model = builtInEmbeddingModels.find((candidate) => candidate.id === configured);
	if (!model) {
		const known = builtInEmbeddingModels.map((candidate) => candidate.id).join(', ');
		throw new Error(`RAG_EMBEDDING_MODEL "${configured}" is not registered (known: ${known}).`);
	}

	return model;
}

export const activeEmbeddingModel: EmbeddingModel = readActiveEmbeddingModel();
