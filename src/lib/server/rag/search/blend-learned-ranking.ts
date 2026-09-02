export interface LearnedRankingCandidate<T> {
	value: T;
	learnedScore: number;
}

export interface BlendedRankingCandidate<T> extends LearnedRankingCandidate<T> {
	blendedScore: number;
	originalIndex: number;
}

function normalizedRankScore(index: number, candidateCount: number): number {
	if (candidateCount <= 1) return 1;
	return 1 - index / (candidateCount - 1);
}

export function blendLearnedRanking<T>(
	candidates: readonly LearnedRankingCandidate<T>[],
	learnedWeight: number
): BlendedRankingCandidate<T>[] {
	if (!Number.isFinite(learnedWeight) || learnedWeight < 0 || learnedWeight > 1) {
		throw new Error('The learned retrieval blend weight must be between zero and one.');
	}

	if (!candidates.every(({ learnedScore }) => Number.isFinite(learnedScore))) {
		throw new Error('Cannot blend a non-finite learned retrieval score.');
	}

	const learnedOrder = candidates
		.map((candidate, originalIndex) => ({ candidate, originalIndex }))
		.sort(
			(left, right) =>
				right.candidate.learnedScore - left.candidate.learnedScore ||
				left.originalIndex - right.originalIndex
		);
	const learnedIndexes = new Map(
		learnedOrder.map(({ originalIndex }, learnedIndex) => [originalIndex, learnedIndex])
	);
	const baseWeight = 1 - learnedWeight;

	return candidates
		.map((candidate, originalIndex) => {
			const learnedIndex = learnedIndexes.get(originalIndex);
			if (learnedIndex === undefined) {
				throw new Error('A learned retrieval candidate is missing from its ranked order.');
			}

			const baseRankScore = normalizedRankScore(originalIndex, candidates.length);
			const learnedRankScore = normalizedRankScore(learnedIndex, candidates.length);

			return {
				...candidate,
				blendedScore: baseWeight * baseRankScore + learnedWeight * learnedRankScore,
				originalIndex
			};
		})
		.sort(
			(left, right) =>
				right.blendedScore - left.blendedScore || left.originalIndex - right.originalIndex
		);
}
