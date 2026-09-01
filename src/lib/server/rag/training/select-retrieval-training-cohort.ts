import { EMBEDDING_MODEL } from '$lib/server/rag/embedding-model';
import {
	CROSS_ENCODER_MODEL,
	RETRIEVAL_SCORING_VERSION
} from '$lib/server/rag/search/retrieval-version';
import {
	MINIMUM_DISTINCT_QUERIES,
	MINIMUM_TRAINING_EXAMPLES
} from './retrieval-training-constants';
import type { RetrievalTrainingExample } from './retrieval-training.types';

export interface RetrievalTrainingCohort {
	examples: RetrievalTrainingExample[];
	embeddingModel: string;
	rerankerModel: string;
	scoringVersion: string;
	distinctQueries: number;
}

export class RetrievalTrainingReadinessError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'RetrievalTrainingReadinessError';
	}
}

export function selectRetrievalTrainingCohort(
	examples: RetrievalTrainingExample[]
): RetrievalTrainingCohort {
	const compatibleExamples = examples.filter(
		(example) =>
			example.embeddingModel === EMBEDDING_MODEL &&
			example.rerankerModel === CROSS_ENCODER_MODEL &&
			example.scoringVersion === RETRIEVAL_SCORING_VERSION
	);

	const distinctQueries = new Set(compatibleExamples.map((example) => example.queryHash)).size;

	if (compatibleExamples.length < MINIMUM_TRAINING_EXAMPLES) {
		throw new RetrievalTrainingReadinessError(
			`Retrieval training needs at least ${MINIMUM_TRAINING_EXAMPLES} compatible ratings; ` +
				`only ${compatibleExamples.length} are currently available.`
		);
	}

	if (distinctQueries < MINIMUM_DISTINCT_QUERIES) {
		throw new RetrievalTrainingReadinessError(
			`Retrieval training needs ratings from at least ${MINIMUM_DISTINCT_QUERIES} distinct ` +
				`queries; only ${distinctQueries} are currently available.`
		);
	}

	return {
		examples: compatibleExamples,
		embeddingModel: EMBEDDING_MODEL,
		rerankerModel: CROSS_ENCODER_MODEL,
		scoringVersion: RETRIEVAL_SCORING_VERSION,
		distinctQueries
	};
}
