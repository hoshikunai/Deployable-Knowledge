import type { PreparedRetrievalTrainingExample } from './build-retrieval-training-features';

export interface RetrievalPreferencePair {
	preferredFeedbackId: string;
	rejectedFeedbackId: string;
	featureDifferences: number[];
	ratingGap: number;
	weight: number;
}

interface UnweightedRetrievalPreferencePair extends Omit<RetrievalPreferencePair, 'weight'> {
	rawWeight: number;
}

function buildGroupKey(example: PreparedRetrievalTrainingExample): string {
	return `${example.impressionId}\u0000${example.retrievalMode}`;
}

function subtractFeatures(preferred: number[], rejected: number[]): number[] {
	if (preferred.length !== rejected.length) {
		throw new Error('Cannot compare retrieval examples with different feature dimensions.');
	}

	const differences = preferred.map((value, index) => value - rejected[index]);
	if (!differences.every(Number.isFinite)) {
		throw new Error('A retrieval preference pair contains a non-finite feature difference.');
	}

	return differences;
}

export function buildRetrievalPreferencePairs(
	examples: PreparedRetrievalTrainingExample[]
): RetrievalPreferencePair[] {
	const groups = new Map<string, PreparedRetrievalTrainingExample[]>();

	for (const example of examples) {
		const groupKey = buildGroupKey(example);
		const group = groups.get(groupKey) ?? [];
		group.push(example);
		groups.set(groupKey, group);
	}

	const pairs: RetrievalPreferencePair[] = [];

	for (const group of groups.values()) {
		const groupPairs: UnweightedRetrievalPreferencePair[] = [];

		for (let leftIndex = 0; leftIndex < group.length; leftIndex += 1) {
			for (let rightIndex = leftIndex + 1; rightIndex < group.length; rightIndex += 1) {
				const left = group[leftIndex];
				const right = group[rightIndex];
				if (left.rating === right.rating) continue;

				const preferred = left.rating > right.rating ? left : right;
				const rejected = left.rating > right.rating ? right : left;
				const ratingGap = preferred.rating - rejected.rating;

				groupPairs.push({
					preferredFeedbackId: preferred.feedbackId,
					rejectedFeedbackId: rejected.feedbackId,
					featureDifferences: subtractFeatures(preferred.features, rejected.features),
					ratingGap,
					rawWeight: ratingGap
				});
			}
		}

		const groupWeight = groupPairs.reduce((total, pair) => total + pair.rawWeight, 0);
		if (groupWeight === 0) continue;

		pairs.push(
			...groupPairs.map(({ rawWeight, ...pair }) => ({
				...pair,
				weight: rawWeight / groupWeight
			}))
		);
	}

	return pairs;
}
