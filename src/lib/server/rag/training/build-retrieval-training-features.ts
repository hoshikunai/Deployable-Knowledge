import { RetrievalMode } from '$lib/enums';
import type { RetrievalTrainingExample } from './retrieval-training.types';

export const RETRIEVAL_FEATURE_NAMES = [
	'baseScore',
	'semanticScore',
	'semanticScorePresent',
	'bm25Score',
	'bm25ScorePresent',
	'crossEncoderScore',
	'crossEncoderScorePresent',
	'reciprocalBaseRank',
	'modeSemantic',
	'modeBm25',
	'modeHybrid'
] as const;

export interface RetrievalFeatureInput {
	retrievalMode: RetrievalMode;
	baseRank: number;
	semanticScore: number | null;
	bm25Score: number | null;
	crossEncoderScore: number | null;
	baseScore: number;
}

export interface PreparedRetrievalTrainingExample {
	feedbackId: string;
	impressionId: string;
	impressionResultId: string;
	queryHash: string;
	retrievalMode: RetrievalMode;
	rating: RetrievalTrainingExample['rating'];
	baseRank: number;
	features: number[];
}

export function buildRetrievalFeatureVector(input: RetrievalFeatureInput): number[] {
	const features = [
		input.baseScore,
		input.semanticScore ?? 0,
		Number(input.semanticScore !== null),
		input.bm25Score ?? 0,
		Number(input.bm25Score !== null),
		input.crossEncoderScore ?? 0,
		Number(input.crossEncoderScore !== null),
		1 / Math.max(input.baseRank, 1),
		Number(input.retrievalMode === RetrievalMode.SEMANTIC),
		Number(input.retrievalMode === RetrievalMode.BM25),
		Number(input.retrievalMode === RetrievalMode.HYBRID)
	];

	if (!features.every(Number.isFinite)) {
		throw new Error('Retrieval feature vector contains a non-finite value.');
	}

	return features;
}

export function buildRetrievalTrainingFeatures(
	examples: RetrievalTrainingExample[]
): PreparedRetrievalTrainingExample[] {
	return examples.map((example) => ({
		feedbackId: example.feedbackId,
		impressionId: example.impressionId,
		impressionResultId: example.impressionResultId,
		queryHash: example.queryHash,
		retrievalMode: example.retrievalMode,
		rating: example.rating,
		baseRank: example.baseRank,
		features: buildRetrievalFeatureVector(example)
	}));
}
