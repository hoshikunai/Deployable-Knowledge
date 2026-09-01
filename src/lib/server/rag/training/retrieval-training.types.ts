import type { RetrievalMode } from '$lib/enums';
import type { ChunkRatingValue } from '$lib/types';

export const RETRIEVAL_TRAINING_DATASET_VERSION = 1 as const;

export const RETRIEVAL_TARGET_BY_RATING = {
	1: -1,
	2: -0.5,
	3: 0,
	4: 0.5,
	5: 1
} satisfies Record<ChunkRatingValue, number>;

export interface RetrievalTrainingExample {
	feedbackId: string;
	impressionId: string;
	impressionResultId: string;
	chunkId: string;
	queryHash: string;
	rating: ChunkRatingValue;
	target: number;
	retrievalMode: RetrievalMode;
	baseRank: number;
	displayedRank: number;
	semanticScore: number | null;
	bm25Score: number | null;
	crossEncoderScore: number | null;
	baseScore: number;
	requestedTopK: number;
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	impressionCreatedAt: string;
	feedbackUpdatedAt: string;
}

export interface RetrievalTrainingDatasetStats {
	totalFeedback: number;
	attributedFeedback: number;
	unattributedFeedback: number;
	inconsistentFeedback: number;
	distinctQueries: number;
	ratingCounts: Record<ChunkRatingValue, number>;
	modeCounts: Record<RetrievalMode, number>;
	scoreAvailability: {
		semantic: number;
		bm25: number;
		crossEncoder: number;
	};
}

export interface RetrievalTrainingDataset {
	version: typeof RETRIEVAL_TRAINING_DATASET_VERSION;
	generatedAt: string;
	examples: RetrievalTrainingExample[];
	stats: RetrievalTrainingDatasetStats;
}
