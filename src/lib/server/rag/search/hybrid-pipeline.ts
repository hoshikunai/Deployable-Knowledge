import { CANDIDATE_DEDUP_MODES, type CandidateDedupMode } from './candidate-dedup';
import { findCrossEncoder, getActiveCrossEncoder } from './cross-encoders/registry';
import { DOCUMENT_ORDERS, type DocumentOrder } from './document-scoring';
import { RRF_FUSION_LEVELS, type RrfFusionLevel } from './rrf-fusion';

export type HybridFusion = 'union' | 'rrf';

export const HYBRID_FUSIONS: readonly HybridFusion[] = ['union', 'rrf'];

export type HybridPipelineConfig = {
	fusion: HybridFusion;
	reranker: string | null;
	retrievalMultiplier: number;
	rerankMultiplier: number;
	rrfRankConstant: number;
	rerankMaxTokens: number;
	dedup: CandidateDedupMode;
	maxChunksPerDocument: number;
	rrfSemanticWeight: number;
	rrfBm25Weight: number;
	fusionLevel: RrfFusionLevel;
	documentOrder: DocumentOrder;
	rerankTitle: boolean;
};

const DEFAULT_HYBRID_PIPELINE: Omit<HybridPipelineConfig, 'reranker'> = {
	fusion: 'union',
	retrievalMultiplier: 2,
	rerankMultiplier: 2,
	rrfRankConstant: 60,
	rerankMaxTokens: 512,
	dedup: 'none',
	maxChunksPerDocument: 2,
	rrfSemanticWeight: 1,
	rrfBm25Weight: 1,
	fusionLevel: 'chunk',
	documentOrder: 'none',
	rerankTitle: false
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
	if (!raw) return getActiveCrossEncoder().id;
	if (raw === NO_RERANKER) return null;

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

function readRrfWeights(): Pick<HybridPipelineConfig, 'rrfSemanticWeight' | 'rrfBm25Weight'> {
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
		fusion: readChoice('RAG_HYBRID_FUSION', HYBRID_FUSIONS, DEFAULT_HYBRID_PIPELINE.fusion),
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
		rerankTitle:
			readChoice(
				'RAG_RERANK_TITLE',
				['true', 'false'] as const,
				DEFAULT_HYBRID_PIPELINE.rerankTitle ? 'true' : 'false'
			) === 'true'
	};
}

let hybridPipeline: Readonly<HybridPipelineConfig> | undefined;

export function getHybridPipeline(): Readonly<HybridPipelineConfig> {
	hybridPipeline ??= Object.freeze(readHybridPipelineConfig());
	return hybridPipeline;
}
