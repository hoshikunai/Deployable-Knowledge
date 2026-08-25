import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { describe, it } from 'node:test';
import type { GraphSearchResult } from '../../src/lib/server/knowledge-graph-new/search';
import { evaluateAssertions, type EvaluatedGraphAssertion } from './assertion-evaluator';
import type { GoldBenchmark } from './benchmark-types';
import { validateCorpusChunks, validateGoldBenchmark } from './benchmark-validation';
import { evaluateRetrieval } from './retrieval-evaluator';

const content =
	'A casualty receiving and treatment ship (CRTS) is part of an amphibious ready group (ARG).';

const benchmark: GoldBenchmark = {
	version: 'test-v1',
	corpusId: 'test-corpus',
	canonicalRelations: ['is_part_of', 'uses'],
	chunks: [
		{
			chunkId: 'chunk-1',
			documentId: 'document-1',
			contentSha256: createHash('sha256').update(content).digest('hex')
		},
		{
			chunkId: 'chunk-2',
			documentId: 'document-1',
			contentSha256: createHash('sha256').update('irrelevant').digest('hex')
		}
	],
	assertions: [
		{
			id: 'gold-1',
			documentId: 'document-1',
			chunkId: 'chunk-1',
			subject: {
				canonical: 'casualty receiving and treatment ship',
				aliases: ['CRTS'],
				type: 'object'
			},
			predicate: 'is_part_of',
			object: { canonical: 'amphibious ready group', aliases: ['ARG'], type: 'organization' },
			evidence: content,
			status: 'asserted',
			required: true
		}
	],
	queries: [
		{
			id: 'query-1',
			question: 'What organization is CRTS part of?',
			relevantChunkIds: ['chunk-1'],
			expectedPathAssertionIds: [['gold-1']],
			forbiddenChunkIds: ['chunk-2']
		}
	]
};

describe('gold benchmark validation', () => {
	it('validates IDs, hashes, references, and verbatim evidence', () => {
		assert.deepEqual(validateGoldBenchmark(benchmark), []);
		assert.deepEqual(
			validateCorpusChunks(benchmark, [
				{ chunkId: 'chunk-1', documentId: 'document-1', content },
				{ chunkId: 'chunk-2', documentId: 'document-1', content: 'irrelevant' }
			]),
			[]
		);
	});

	it('rejects placeholder IDs and changed chunks', () => {
		const invalid = structuredClone(benchmark);
		invalid.chunks[0].chunkId = 'replace-with-real-chunk-id';
		assert.ok(validateGoldBenchmark(invalid).some((error) => error.includes('placeholder')));
		assert.ok(
			validateCorpusChunks(benchmark, [
				{ chunkId: 'chunk-1', documentId: 'document-1', content: 'changed' }
			]).some((error) => error.includes('SHA-256 mismatch'))
		);
	});
});

describe('assertion evaluation', () => {
	it('matches aliases without losing direction, type, or modality checks', () => {
		const actual: EvaluatedGraphAssertion[] = [
			{
				id: 'actual-1',
				documentId: 'document-1',
				chunkId: 'chunk-1',
				subject: 'CRTS',
				subjectType: 'object',
				canonicalPredicate: 'is_part_of',
				object: 'ARG',
				objectType: 'organization',
				status: 'asserted'
			}
		];
		const result = evaluateAssertions(benchmark, actual);
		assert.equal(result.precision, 1);
		assert.equal(result.recall, 1);
		assert.equal(result.directionAccuracy, 1);
		assert.equal(result.endpointTypeAccuracy, 1);
		assert.equal(result.modalityAccuracy, 1);
	});

	it('reports a reversed edge as both an error and a missed required assertion', () => {
		const actual: EvaluatedGraphAssertion[] = [
			{
				id: 'actual-reversed',
				documentId: 'document-1',
				chunkId: 'chunk-1',
				subject: 'ARG',
				subjectType: 'organization',
				canonicalPredicate: 'is_part_of',
				object: 'CRTS',
				objectType: 'object',
				status: 'asserted'
			}
		];
		const result = evaluateAssertions(benchmark, actual);
		assert.equal(result.precision, 0);
		assert.equal(result.recall, 0);
		assert.equal(result.reversedAssertionCount, 1);
		assert.equal(result.directionAccuracy, 0);
	});
});

describe('retrieval and graph-path evaluation', () => {
	it('calculates ranked retrieval and strict ordered path metrics', () => {
		const results = new Map<string, GraphSearchResult>([
			[
				'query-1',
				{
					chunks: [
						{ chunkId: 'chunk-2', score: 1, supportingAssertionIds: [] },
						{ chunkId: 'chunk-1', score: 0.5, supportingAssertionIds: ['actual-1'] }
					],
					paths: [{ assertionIds: ['actual-1'], score: 1 }]
				}
			]
		]);
		const evaluation = evaluateRetrieval(
			benchmark,
			results,
			[
				{
					actualAssertionId: 'actual-1',
					goldAssertionId: 'gold-1',
					required: true,
					typesCorrect: true,
					statusCorrect: true
				}
			],
			[1, 2]
		);
		assert.equal(evaluation.meanReciprocalRank, 0.5);
		assert.equal(evaluation.metricsAtK['1'].recall, 0);
		assert.equal(evaluation.metricsAtK['1'].forbiddenHit, 1);
		assert.equal(evaluation.metricsAtK['2'].recall, 1);
		assert.equal(evaluation.meanPathRecall, 1);
	});
});
