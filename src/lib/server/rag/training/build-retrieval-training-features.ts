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

export interface PreparedRetrievalTrainingExample {
	feedbackId: string;
	queryHash: string;
	retrievalMode: RetrievalMode;
	rating: RetrievalTrainingExample['rating'];
	target: number;
	baseRank: number;
	features: number[];
}

export function buildRetrievalTrainingFeatures(
	examples: RetrievalTrainingExample[]
): PreparedRetrievalTrainingExample[] {
	return examples.map((example) => {
		const features = [
			example.baseScore,
			example.semanticScore ?? 0,
			Number(example.semanticScore !== null),
			example.bm25Score ?? 0,
			Number(example.bm25Score !== null),
			example.crossEncoderScore ?? 0,
			Number(example.crossEncoderScore !== null),
			1 / Math.max(example.baseRank, 1),
			Number(example.retrievalMode === RetrievalMode.SEMANTIC),
			Number(example.retrievalMode === RetrievalMode.BM25),
			Number(example.retrievalMode === RetrievalMode.HYBRID)
		];

		if (!features.every(Number.isFinite)) {
			throw new Error(`Training example ${example.feedbackId} contains a non-finite feature.`);
		}

		return {
			feedbackId: example.feedbackId,
			queryHash: example.queryHash,
			retrievalMode: example.retrievalMode,
			rating: example.rating,
			target: example.target,
			baseRank: example.baseRank,
			features
		};
	});
}
