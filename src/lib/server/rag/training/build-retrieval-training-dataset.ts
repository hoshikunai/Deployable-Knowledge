import { CHUNK_RATING_VALUES, HUMAN_EXPERT_FEEDBACK_SOURCE } from '$lib/constants';
import { RetrievalMode } from '$lib/enums';
import { RetrievalTrainingRepository } from '$lib/server/repositories';
import type { ChunkRatingValue } from '$lib/types';
import {
	RETRIEVAL_TRAINING_DATASET_VERSION,
	type RetrievalTrainingCandidate,
	type RetrievalTrainingDataset,
	type RetrievalTrainingExample
} from './retrieval-training.types';

const RETRIEVAL_MODES = new Set<string>(Object.values(RetrievalMode));

function retrievalGroupKey(impressionId: string, retrievalMode: RetrievalMode): string {
	return `${impressionId}\u0000${retrievalMode}`;
}

export async function buildRetrievalTrainingDataset(): Promise<RetrievalTrainingDataset> {
	const feedbackSource = HUMAN_EXPERT_FEEDBACK_SOURCE;
	const rows = await RetrievalTrainingRepository.readDatasetRows();
	const impressionIds = [
		...new Set(
			rows.flatMap((row) => {
				if (row.impressionId === null) return [];
				return [row.impressionId];
			})
		)
	];
	const candidateRows = await RetrievalTrainingRepository.readCandidateRows(impressionIds);
	const candidateGroups = new Map<string, RetrievalTrainingCandidate[]>();

	for (const candidate of candidateRows) {
		if (!RETRIEVAL_MODES.has(candidate.retrievalMode)) continue;

		const retrievalMode = candidate.retrievalMode as RetrievalMode;
		const key = retrievalGroupKey(candidate.impressionId, retrievalMode);
		const group = candidateGroups.get(key) ?? [];

		group.push({
			impressionResultId: candidate.impressionResultId,
			retrievalMode,
			baseRank: candidate.baseRank,
			semanticScore: candidate.semanticScore,
			bm25Score: candidate.bm25Score,
			crossEncoderScore: candidate.crossEncoderScore,
			baseScore: candidate.baseScore
		});

		candidateGroups.set(key, group);
	}

	for (const group of candidateGroups.values()) {
		group.sort((left, right) => left.baseRank - right.baseRank);
	}

	const examples: RetrievalTrainingExample[] = [];
	const ratingCounts: Record<ChunkRatingValue, number> = {
		1: 0,
		2: 0,
		3: 0,
		4: 0,
		5: 0
	};
	const modeCounts: Record<RetrievalMode, number> = {
		[RetrievalMode.SEMANTIC]: 0,
		[RetrievalMode.BM25]: 0,
		[RetrievalMode.HYBRID]: 0
	};
	const scoreAvailability = {
		semantic: 0,
		bm25: 0,
		crossEncoder: 0
	};
	const queryHashes = new Set<string>();
	let unattributedFeedback = 0;
	let inconsistentFeedback = 0;

	for (const row of rows) {
		if (!CHUNK_RATING_VALUES.includes(row.rating as ChunkRatingValue)) {
			inconsistentFeedback += 1;
			continue;
		}

		const rating = row.rating as ChunkRatingValue;
		ratingCounts[rating] += 1;

		if (
			row.feedbackImpressionResultId === null ||
			row.resultId === null ||
			row.resultChunkId === null ||
			row.resultRetrievalMode === null ||
			row.baseRank === null ||
			row.displayedRank === null ||
			row.baseScore === null ||
			row.impressionId === null ||
			row.requestedTopK === null ||
			row.embeddingModel === null ||
			row.rerankerModel === null ||
			row.scoringVersion === null ||
			row.impressionCreatedAt === null
		) {
			unattributedFeedback += 1;
			continue;
		}

		if (
			row.feedbackImpressionResultId !== row.resultId ||
			row.feedbackChunkId !== row.resultChunkId ||
			row.feedbackRetrievalMode !== row.resultRetrievalMode ||
			row.feedbackResultRank !== row.displayedRank ||
			!RETRIEVAL_MODES.has(row.resultRetrievalMode)
		) {
			inconsistentFeedback += 1;
			continue;
		}

		const retrievalMode = row.resultRetrievalMode as RetrievalMode;
		const candidateGroup = candidateGroups.get(retrievalGroupKey(row.impressionId, retrievalMode));

		if (
			!candidateGroup ||
			!candidateGroup.some((candidate) => candidate.impressionResultId === row.resultId)
		) {
			inconsistentFeedback += 1;
			continue;
		}

		examples.push({
			feedbackId: row.feedbackId,
			impressionId: row.impressionId,
			impressionResultId: row.resultId,
			chunkId: row.resultChunkId,
			queryHash: row.queryHash,
			rating,
			feedbackSource: row.feedbackSource,
			feedbackConfidence: row.feedbackConfidence,
			feedbackRationale: row.feedbackRationale,
			retrievalMode,
			baseRank: row.baseRank,
			displayedRank: row.displayedRank,
			semanticScore: row.semanticScore,
			bm25Score: row.bm25Score,
			crossEncoderScore: row.crossEncoderScore,
			baseScore: row.baseScore,
			requestedTopK: row.requestedTopK,
			embeddingModel: row.embeddingModel,
			rerankerModel: row.rerankerModel,
			scoringVersion: row.scoringVersion,
			impressionCreatedAt: row.impressionCreatedAt,
			feedbackUpdatedAt: row.feedbackUpdatedAt,
			candidateGroup
		});

		queryHashes.add(row.queryHash);
		modeCounts[retrievalMode] += 1;

		if (row.semanticScore !== null) scoreAvailability.semantic += 1;
		if (row.bm25Score !== null) scoreAvailability.bm25 += 1;
		if (row.crossEncoderScore !== null) scoreAvailability.crossEncoder += 1;
	}

	return {
		version: RETRIEVAL_TRAINING_DATASET_VERSION,
		feedbackSource,
		generatedAt: new Date().toISOString(),
		examples,
		stats: {
			feedbackSource,
			totalFeedback: rows.length,
			attributedFeedback: examples.length,
			unattributedFeedback,
			inconsistentFeedback,
			distinctQueries: queryHashes.size,
			ratingCounts,
			modeCounts,
			scoreAvailability
		}
	};
}
