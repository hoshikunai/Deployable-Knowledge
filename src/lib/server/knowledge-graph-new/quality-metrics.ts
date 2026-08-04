export const KNOWLEDGE_GRAPH_TARGETS = {
	usefulTriplePrecision: 0.85,
	directionAccuracy: 0.9,
	endpointTypeAccuracy: 0.9,
	canonicalRelationCoverage: 0.9
} as const;

export type ExtractionSource = 'llm-only' | 'gliner-only' | 'agreement';

export interface ReviewedAssertion {
	assertionId: string;
	documentId: string;
	chunkId: string;
	source: ExtractionSource;

	// Whether this assertion would be written to the graph.
	accepted: boolean;

	// Manually reviewed labels.
	useful: boolean;
	directionCorrect: boolean;
	endpointTypesCorrect: boolean;

	// null means the predicate could not be mapped.
	canonicalRelation: string | null;
}

export interface BenchmarkReview {
	expectedUsefulAssertionCount: number;
	recoveredUsefulAssertionCount: number;
	assertions: ReviewedAssertion[];
}

export interface QualityMetrics {
	reviewedCandidateCount: number;
	acceptedAssertionCount: number;
	acceptanceRate: number;
	usefulTriplePrecision: number;
	directionAccuracy: number;
	endpointTypeAccuracy: number;
	canonicalRelationCoverage: number;
	usefulAssertionRecall: number;
}

export function calculateQualityMetrics(review: BenchmarkReview): QualityMetrics {
	validateReview(review);

	const accepted = review.assertions.filter((assertion) => assertion.accepted);

	return {
		reviewedCandidateCount: review.assertions.length,
		acceptedAssertionCount: accepted.length,
		acceptanceRate: ratio(accepted.length, review.assertions.length),
		usefulTriplePrecision: ratio(
			accepted.filter((assertion) => assertion.useful).length,
			accepted.length
		),
		directionAccuracy: ratio(
			accepted.filter((assertion) => assertion.directionCorrect).length,
			accepted.length
		),
		endpointTypeAccuracy: ratio(
			accepted.filter((assertion) => assertion.endpointTypesCorrect).length,
			accepted.length
		),
		canonicalRelationCoverage: ratio(
			accepted.filter((assertion) => assertion.canonicalRelation !== null).length,
			accepted.length
		),
		usefulAssertionRecall: ratio(
			review.recoveredUsefulAssertionCount,
			review.expectedUsefulAssertionCount
		)
	};
}

export function qualityTargetFailures(metrics: QualityMetrics): string[] {
	const failures: string[] = [];

	if (metrics.usefulTriplePrecision < KNOWLEDGE_GRAPH_TARGETS.usefulTriplePrecision) {
		failures.push(`Useful-triple precision: ${percent(metrics.usefulTriplePrecision)}`);
	}

	if (metrics.directionAccuracy < KNOWLEDGE_GRAPH_TARGETS.directionAccuracy) {
		failures.push(`Direction accuracy: ${percent(metrics.directionAccuracy)}`);
	}

	if (metrics.endpointTypeAccuracy < KNOWLEDGE_GRAPH_TARGETS.endpointTypeAccuracy) {
		failures.push(`Endpoint-type accuracy: ${percent(metrics.endpointTypeAccuracy)}`);
	}

	if (metrics.canonicalRelationCoverage < KNOWLEDGE_GRAPH_TARGETS.canonicalRelationCoverage) {
		failures.push(`Canonical-relation coverage: ${percent(metrics.canonicalRelationCoverage)}`);
	}

	return failures;
}

function ratio(numerator: number, denominator: number): number {
	if (denominator === 0) return 0;
	return numerator / denominator;
}

function percent(value: number): string {
	return `${(value * 100).toFixed(1)}%`;
}

function validateReview(review: BenchmarkReview): void {
	if (review.expectedUsefulAssertionCount < 0) {
		throw new Error('Expected assertion count cannot be negative.');
	}

	if (
		review.recoveredUsefulAssertionCount < 0 ||
		review.recoveredUsefulAssertionCount > review.expectedUsefulAssertionCount
	) {
		throw new Error('Recovered assertion count is outside the expected range.');
	}
}
