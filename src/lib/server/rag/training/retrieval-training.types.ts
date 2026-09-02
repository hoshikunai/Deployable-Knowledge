import type { RetrievalMode } from '$lib/enums';
import type { ChunkRatingValue, RetrievalFeedbackSource } from '$lib/types';

export const RETRIEVAL_TRAINING_DATASET_VERSION = 2 as const;

export interface RetrievalTrainingExample {
	feedbackId: string;
	impressionId: string;
	impressionResultId: string;
	chunkId: string;
	queryHash: string;
	rating: ChunkRatingValue;
	feedbackSource: RetrievalFeedbackSource;
	feedbackConfidence: number | null;
	feedbackRationale: string | null;
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
	feedbackSource: RetrievalFeedbackSource;
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
	feedbackSource: RetrievalFeedbackSource;
	generatedAt: string;
	examples: RetrievalTrainingExample[];
	stats: RetrievalTrainingDatasetStats;
}
