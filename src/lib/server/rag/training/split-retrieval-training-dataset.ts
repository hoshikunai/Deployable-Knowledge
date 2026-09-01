import { VALIDATION_FRACTION } from './retrieval-training-constants';
import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import { RetrievalTrainingReadinessError } from './select-retrieval-training-cohort';

export interface RetrievalTrainingSplit {
	training: PreparedRetrievalTrainingExample[];
	validation: PreparedRetrievalTrainingExample[];
}

export function splitRetrievalTrainingDataset(
	examples: PreparedRetrievalTrainingExample[]
): RetrievalTrainingSplit {
	const queryHashes = [...new Set(examples.map((example) => example.queryHash))].sort();

	if (queryHashes.length < 2) {
		throw new RetrievalTrainingReadinessError(
			'Retrieval training requires at least two distinct queries for validation.'
		);
	}

	const validationQueryCount = Math.max(1, Math.floor(queryHashes.length * VALIDATION_FRACTION));
	const validationQueryHashes = new Set(queryHashes.slice(0, validationQueryCount));

	const training = examples.filter((example) => !validationQueryHashes.has(example.queryHash));
	const validation = examples.filter((example) => validationQueryHashes.has(example.queryHash));

	if (training.length === 0 || validation.length === 0) {
		throw new RetrievalTrainingReadinessError(
			'The retrieval dataset could not be divided into training and validation partitions.'
		);
	}

	return {
		training,
		validation
	};
}
