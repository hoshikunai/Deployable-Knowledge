import { CROSS_VALIDATION_FOLD_COUNT } from './retrieval-training-constants';
import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';
import { RetrievalTrainingReadinessError } from './select-retrieval-training-cohort';

export interface RetrievalCrossValidationFold {
	index: number;
	training: PreparedRetrievalTrainingExample[];
	validation: PreparedRetrievalTrainingExample[];
}

export function buildQueryGroupedCrossValidationFolds(
	examples: PreparedRetrievalTrainingExample[],
	foldCount: number = CROSS_VALIDATION_FOLD_COUNT
): RetrievalCrossValidationFold[] {
	const queryExampleCounts = new Map<string, number>();
	for (const example of examples) {
		queryExampleCounts.set(example.queryHash, (queryExampleCounts.get(example.queryHash) ?? 0) + 1);
	}

	const queryGroups = [...queryExampleCounts]
		.map(([queryHash, exampleCount]) => ({ queryHash, exampleCount }))
		.sort(
			(left, right) =>
				right.exampleCount - left.exampleCount || left.queryHash.localeCompare(right.queryHash)
		);

	if (!Number.isInteger(foldCount) || foldCount < 2) {
		throw new Error('Retrieval cross-validation requires at least two folds.');
	}

	if (queryGroups.length < foldCount) {
		throw new RetrievalTrainingReadinessError(
			`Retrieval cross-validation requires at least ${foldCount} distinct queries; ` +
				`only ${queryGroups.length} are currently available.`
		);
	}

	const foldAssignments = Array.from({ length: foldCount }, () => ({
		queryHashes: new Set<string>(),
		exampleCount: 0
	}));

	for (const queryGroup of queryGroups) {
		const assignment = foldAssignments.reduce((best, candidate) => {
			if (candidate.exampleCount !== best.exampleCount) {
				return candidate.exampleCount < best.exampleCount ? candidate : best;
			}

			return candidate.queryHashes.size < best.queryHashes.size ? candidate : best;
		});

		assignment.queryHashes.add(queryGroup.queryHash);
		assignment.exampleCount += queryGroup.exampleCount;
	}

	return foldAssignments.map((assignment, foldIndex) => {
		const validationQueryHashes = assignment.queryHashes;
		const training = examples.filter((example) => !validationQueryHashes.has(example.queryHash));
		const validation = examples.filter((example) => validationQueryHashes.has(example.queryHash));

		if (training.length === 0 || validation.length === 0) {
			throw new RetrievalTrainingReadinessError(
				`Retrieval cross-validation fold ${foldIndex + 1} does not contain both partitions.`
			);
		}

		return {
			index: foldIndex,
			training,
			validation
		};
	});
}
