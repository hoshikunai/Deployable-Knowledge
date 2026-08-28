import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	calculateQualityMetrics,
	type BenchmarkReview
} from '../../src/lib/server/knowledge-graph-new/quality-metrics';

const reviewedAssertions: BenchmarkReview['assertions'] = [
	{
		assertionId: 'assertion-1',
		documentId: 'document-1',
		chunkId: 'chunk-1',
		source: 'llm-only',
		accepted: true,
		useful: true,
		directionCorrect: true,
		endpointTypesCorrect: true,
		canonicalRelation: 'uses'
	}
];

describe('knowledge graph quality metrics', () => {
	it('calculates reviewed-sample quality without inventing recall', () => {
		const metrics = calculateQualityMetrics({ assertions: reviewedAssertions });

		assert.equal(metrics.usefulTriplePrecision, 1);
		assert.equal(metrics.directionAccuracy, 1);
		assert.equal(metrics.endpointTypeAccuracy, 1);
		assert.equal(metrics.canonicalRelationCoverage, 1);
		assert.equal(metrics.usefulAssertionRecall, null);
	});

	it('calculates recall when a complete gold inventory is supplied', () => {
		const metrics = calculateQualityMetrics({
			expectedUsefulAssertionCount: 2,
			recoveredUsefulAssertionCount: 1,
			assertions: reviewedAssertions
		});

		assert.equal(metrics.usefulAssertionRecall, 0.5);
	});

	it('rejects incomplete recall counts', () => {
		assert.throws(
			() =>
				calculateQualityMetrics({
					expectedUsefulAssertionCount: 1,
					assertions: reviewedAssertions
				}),
			/Expected and recovered assertion counts must be supplied together/
		);
	});
});
