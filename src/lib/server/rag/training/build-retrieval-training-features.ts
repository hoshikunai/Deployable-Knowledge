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
	'modeHybrid',
	'baseScoreNormalized',
	'semanticScoreNormalized',
	'bm25ScoreNormalized',
	'crossEncoderScoreNormalized',
	'reciprocalSemanticRank',
	'reciprocalBm25Rank',
	'reciprocalCrossEncoderRank',
	'semanticBm25RankAgreement',
	'retrieverCoverage'
] as const;

export const RETRIEVAL_UNSCALED_FEATURE_NAMES = new Set<string>([
	'semanticScorePresent',
	'bm25ScorePresent',
	'crossEncoderScorePresent',
	'modeSemantic',
	'modeBm25',
	'modeHybrid'
]);

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

type ScoreName = 'baseScore' | 'semanticScore' | 'bm25Score' | 'crossEncoderScore';

interface ScoreRange {
	minimum: number;
	maximum: number;
}

function buildScoreRange(
	inputs: readonly RetrievalFeatureInput[],
	scoreName: ScoreName
): ScoreRange | null {
	const values: number[] = [];

	for (const input of inputs) {
		const value = input[scoreName];
		if (value !== null) values.push(value);
	}

	if (values.length === 0) return null;

	return {
		minimum: Math.min(...values),
		maximum: Math.max(...values)
	};
}

function normalizeScore(value: number | null, range: ScoreRange | null): number {
	if (value === null || range === null) return 0;
	if (range.maximum === range.minimum) return 0.5;

	return (value - range.minimum) / (range.maximum - range.minimum);
}

function buildScoreRanks(
	inputs: readonly RetrievalFeatureInput[],
	scoreName: ScoreName
): Array<number | null> {
	const ranked: Array<{
		candidateIndex: number;
		score: number;
		baseRank: number;
	}> = [];

	inputs.forEach((input, candidateIndex) => {
		const score = input[scoreName];
		if (score === null) return;

		ranked.push({
			candidateIndex,
			score,
			baseRank: input.baseRank
		});
	});

	ranked.sort((left, right) => right.score - left.score || left.baseRank - right.baseRank);

	const ranks = new Array<number | null>(inputs.length).fill(null);

	ranked.forEach((candidate, rankIndex) => {
		ranks[candidate.candidateIndex] = rankIndex + 1;
	});

	return ranks;
}

function reciprocalRank(rank: number | null): number {
	if (rank === null) return 0;
	return 1 / Math.max(rank, 1);
}

export function buildRetrievalFeatureVectors(inputs: readonly RetrievalFeatureInput[]): number[][] {
	const baseRange = buildScoreRange(inputs, 'baseScore');
	const semanticRange = buildScoreRange(inputs, 'semanticScore');
	const bm25Range = buildScoreRange(inputs, 'bm25Score');
	const crossEncoderRange = buildScoreRange(inputs, 'crossEncoderScore');

	const semanticRanks = buildScoreRanks(inputs, 'semanticScore');
	const bm25Ranks = buildScoreRanks(inputs, 'bm25Score');
	const crossEncoderRanks = buildScoreRanks(inputs, 'crossEncoderScore');

	return inputs.map((input, index) => {
		const semanticRank = semanticRanks[index];
		const bm25Rank = bm25Ranks[index];
		const crossEncoderRank = crossEncoderRanks[index];

		let semanticBm25RankAgreement = 0;
		if (semanticRank !== null && bm25Rank !== null) {
			semanticBm25RankAgreement = 1 / (1 + Math.abs(semanticRank - bm25Rank));
		}

		const retrieverCoverage =
			[input.semanticScore, input.bm25Score, input.crossEncoderScore].filter(
				(score) => score !== null
			).length / 3;

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
			Number(input.retrievalMode === RetrievalMode.HYBRID),
			normalizeScore(input.baseScore, baseRange),
			normalizeScore(input.semanticScore, semanticRange),
			normalizeScore(input.bm25Score, bm25Range),
			normalizeScore(input.crossEncoderScore, crossEncoderRange),
			reciprocalRank(semanticRank),
			reciprocalRank(bm25Rank),
			reciprocalRank(crossEncoderRank),
			semanticBm25RankAgreement,
			retrieverCoverage
		];

		if (!features.every(Number.isFinite)) {
			throw new Error('Retrieval feature vector contains a non-finite value.');
		}

		return features;
	});
}

function retrievalGroupKey(example: RetrievalTrainingExample): string {
	return `${example.impressionId}\u0000${example.retrievalMode}`;
}

export function buildRetrievalTrainingFeatures(
	examples: RetrievalTrainingExample[]
): PreparedRetrievalTrainingExample[] {
	const featuresByGroup = new Map<string, Map<string, number[]>>();

	return examples.map((example) => {
		const groupKey = retrievalGroupKey(example);
		let featuresByCandidate = featuresByGroup.get(groupKey);

		if (!featuresByCandidate) {
			const featureVectors = buildRetrievalFeatureVectors(example.candidateGroup);

			featuresByCandidate = new Map(
				example.candidateGroup.map((candidate, index) => [
					candidate.impressionResultId,
					featureVectors[index]
				])
			);

			featuresByGroup.set(groupKey, featuresByCandidate);
		}

		const features = featuresByCandidate.get(example.impressionResultId);

		if (!features) {
			throw new Error(
				`Training result ${example.impressionResultId} is missing from its candidate group.`
			);
		}

		return {
			feedbackId: example.feedbackId,
			impressionId: example.impressionId,
			impressionResultId: example.impressionResultId,
			queryHash: example.queryHash,
			retrievalMode: example.retrievalMode,
			rating: example.rating,
			baseRank: example.baseRank,
			features
		};
	});
}
