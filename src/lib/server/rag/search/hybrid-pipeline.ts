// Hybrid pipeline stages are read once from the environment so benchmarks can swap
// rerankers and candidate depths without rebuilding. Unset variables keep the defaults.

import { CANDIDATE_DEDUP_MODES, type CandidateDedupMode } from './candidate-dedup';
import { findCrossEncoder } from './cross-encoders/registry';
import { DOCUMENT_ORDERS, type DocumentOrder } from './document-scoring';
import { RRF_FUSION_LEVELS, type RrfFusionLevel } from './rrf-fusion';

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
	// Experimental: drop repeated RRF candidates before the shortlist cut
	dedup: CandidateDedupMode;
	// Chunks allowed per document when dedup is 'document-cap'
	maxChunksPerDocument: number;
	// RRF weight of each retriever's ranks
	rrfSemanticWeight: number;
	rrfBm25Weight: number;
	// 'document' credits agreement when the retrievers find different chunks of one document
	fusionLevel: RrfFusionLevel;
	// 'sum-top2' orders results by document, scoring each by its two best chunks
	documentOrder: DocumentOrder;
	// Prepend the document title to chunks the reranker reads that don't already start with it
	rerankTitle: boolean;
};

const DEFAULT_HYBRID_PIPELINE: HybridPipelineConfig = {
	reranker: null,
	retrievalMultiplier: 5,
	rerankMultiplier: 2,
	rrfRankConstant: 60,
	rerankMaxTokens: 512,
	dedup: 'none',
	maxChunksPerDocument: 2,
	rrfSemanticWeight: 1,
	rrfBm25Weight: 1,
	fusionLevel: 'chunk',
	documentOrder: 'none',
	rerankTitle: true
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

function readNonNegativeNumber(name: string, fallback: number): number {
	const raw = process.env[name]?.trim();
	if (!raw) return fallback;

	const value = Number(raw);
	if (!Number.isFinite(value) || value < 0) {
		throw new Error(`${name} must be a non-negative number, received "${raw}".`);
	}

	return value;
}

function readChoice<T extends string>(name: string, choices: readonly T[], fallback: T): T {
	const raw = process.env[name]?.trim();
	if (!raw) return fallback;

	const choice = choices.find((candidate) => candidate === raw);
	if (!choice) {
		throw new Error(`${name} "${raw}" must be one of: ${choices.join(', ')}.`);
	}

	return choice;
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

function readRrfWeights(): { rrfSemanticWeight: number; rrfBm25Weight: number } {
	const rrfSemanticWeight = readNonNegativeNumber(
		'RAG_RRF_SEMANTIC_WEIGHT',
		DEFAULT_HYBRID_PIPELINE.rrfSemanticWeight
	);
	const rrfBm25Weight = readNonNegativeNumber(
		'RAG_RRF_BM25_WEIGHT',
		DEFAULT_HYBRID_PIPELINE.rrfBm25Weight
	);

	if (rrfSemanticWeight + rrfBm25Weight === 0) {
		throw new Error('RAG_RRF_SEMANTIC_WEIGHT and RAG_RRF_BM25_WEIGHT cannot both be 0.');
	}

	return { rrfSemanticWeight, rrfBm25Weight };
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
		rerankMaxTokens: readRerankMaxTokens(reranker),
		dedup: readChoice('RAG_HYBRID_DEDUP', CANDIDATE_DEDUP_MODES, DEFAULT_HYBRID_PIPELINE.dedup),
		maxChunksPerDocument: readPositiveInteger(
			'RAG_HYBRID_DOC_CAP',
			DEFAULT_HYBRID_PIPELINE.maxChunksPerDocument
		),
		...readRrfWeights(),
		fusionLevel: readChoice(
			'RAG_RRF_LEVEL',
			RRF_FUSION_LEVELS,
			DEFAULT_HYBRID_PIPELINE.fusionLevel
		),
		documentOrder: readChoice(
			'RAG_HYBRID_DOCUMENT_ORDER',
			DOCUMENT_ORDERS,
			DEFAULT_HYBRID_PIPELINE.documentOrder
		),
		rerankTitle: readChoice('RAG_RERANK_TITLE', ['true', 'false'] as const, 'true') === 'true'
	};
}

export const hybridPipeline: Readonly<HybridPipelineConfig> = Object.freeze(
	readHybridPipelineConfig()
);
