// Hybrid pipeline stages are read once from the environment so benchmarks can swap
// rerankers and candidate depths without rebuilding. Unset variables keep the defaults.

import { findCrossEncoder } from './cross-encoders/registry';

export type HybridPipelineConfig = {
	// Cross-encoder ID that reranks the RRF shortlist, or null for RRF order only
	reranker: string | null;
	// Semantic and BM25 each prefetch topK * retrievalMultiplier chunks
	retrievalMultiplier: number;
	// The reranker scores the first topK * rerankMultiplier RRF candidates
	rerankMultiplier: number;
	rrfRankConstant: number;
	// Query + passage tokens the reranker reads before truncating
	rerankMaxTokens: number;
};

const DEFAULT_HYBRID_PIPELINE: HybridPipelineConfig = {
	reranker: null,
	retrievalMultiplier: 5,
	rerankMultiplier: 2,
	rrfRankConstant: 60,
	rerankMaxTokens: 512
};

const NO_RERANKER = 'none';

function readPositiveInteger(name: string, fallback: number): number {
	const raw = process.env[name]?.trim();
	if (!raw) return fallback;

	const value = Number(raw);
	if (!Number.isInteger(value) || value < 1) {
		throw new Error(`${name} must be a positive integer, received "${raw}".`);
	}

	return value;
}

function readReranker(): string | null {
	const raw = process.env.RAG_HYBRID_RERANKER?.trim();
	if (!raw || raw === NO_RERANKER) return DEFAULT_HYBRID_PIPELINE.reranker;

	if (!findCrossEncoder(raw)) {
		throw new Error(`RAG_HYBRID_RERANKER "${raw}" is not a registered cross-encoder.`);
	}

	return raw;
}

function readRerankMaxTokens(reranker: string | null): number {
	const maxTokens = readPositiveInteger(
		'RAG_RERANK_MAX_TOKENS',
		DEFAULT_HYBRID_PIPELINE.rerankMaxTokens
	);
	const crossEncoder = reranker ? findCrossEncoder(reranker) : null;

	if (crossEncoder && maxTokens > crossEncoder.maxSupportedTokens) {
		throw new Error(
			`RAG_RERANK_MAX_TOKENS ${maxTokens} exceeds ${crossEncoder.name}'s limit of ` +
				`${crossEncoder.maxSupportedTokens} tokens.`
		);
	}

	return maxTokens;
}

function readHybridPipelineConfig(): HybridPipelineConfig {
	const reranker = readReranker();

	return {
		reranker,
		retrievalMultiplier: readPositiveInteger(
			'RAG_RETRIEVAL_MULTIPLIER',
			DEFAULT_HYBRID_PIPELINE.retrievalMultiplier
		),
		rerankMultiplier: readPositiveInteger(
			'RAG_RERANK_MULTIPLIER',
			DEFAULT_HYBRID_PIPELINE.rerankMultiplier
		),
		rrfRankConstant: readPositiveInteger('RAG_RRF_K', DEFAULT_HYBRID_PIPELINE.rrfRankConstant),
		rerankMaxTokens: readRerankMaxTokens(reranker)
	};
}

export const hybridPipeline: Readonly<HybridPipelineConfig> = Object.freeze(
	readHybridPipelineConfig()
);
