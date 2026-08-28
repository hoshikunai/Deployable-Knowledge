import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	evaluateSchemaQuality,
	schemaQualityFailureMessage,
	type SchemaDiscoveryProposal
} from '../../src/lib/server/knowledge-graph-new/schema-quality';
import type {
	CorpusSchema,
	SchemaCategory
} from '../../src/lib/server/knowledge-graph-new/extraction';

const entityTypes: SchemaCategory[] = [
	{ name: 'document', description: 'A document', source: 'universal' },
	{ name: 'process', description: 'A process', source: 'universal' },
	{ name: 'object', description: 'An object', source: 'universal' },
	{ name: 'role', description: 'A role', source: 'universal' }
];

const relations: SchemaCategory[] = [
	{
		name: 'governs_procedure',
		description: 'A document governs a process',
		source: 'llm-consolidated',
		subjectTypes: ['document'],
		objectTypes: ['process']
	},
	{
		name: 'uses_equipment',
		description: 'A process uses an object',
		source: 'llm-consolidated',
		subjectTypes: ['process'],
		objectTypes: ['object']
	},
	{
		name: 'performs_process',
		description: 'A role performs a process',
		source: 'llm-consolidated',
		subjectTypes: ['role'],
		objectTypes: ['process']
	}
];

const proposals: SchemaDiscoveryProposal[] = relations.map((relation, index) => ({
	entityTypes: [],
	relationTypes: [{ ...relation, source: `llm-discovery-${index + 1}` }]
}));

function schema(relationTypes: SchemaCategory[] = relations): CorpusSchema {
	return {
		entityTypes,
		relationTypes,
		sampledChunkIds: ['chunk-1', 'chunk-2', 'chunk-3'],
		version: 'test'
	};
}

describe('adaptive schema quality gate', () => {
	it('passes a closed, distinct schema that represents every discovery batch', () => {
		const report = evaluateSchemaQuality(schema(), {
			discoveryProposals: proposals,
			entityConsolidation: 'complete',
			relationConsolidation: 'complete',
			minimumScore: 0.8
		});

		assert.equal(report.status, 'passed');
		assert.equal(report.score, 1);
		assert.deepEqual(report.issues, []);
	});

	it('fails when consolidation collapses most discovered relation concepts', () => {
		const report = evaluateSchemaQuality(schema([relations[0]]), {
			discoveryProposals: proposals,
			entityConsolidation: 'complete',
			relationConsolidation: 'complete',
			minimumScore: 0.8
		});

		assert.equal(report.status, 'failed');
		assert.equal(report.metrics.candidateCoverage, 0.3333);
		assert.equal(report.metrics.discoveryBatchCoverage, 0.3333);
		assert.ok(report.issues.some((issue) => issue.code === 'score-below-threshold'));
		assert.match(schemaQualityFailureMessage(report), /Schema quality gate failed/);
	});

	it('hard-fails non-closed and provenance-oriented relations', () => {
		const recordedIn: SchemaCategory = {
			name: 'is_recorded_in',
			description: 'A process is recorded in a document',
			source: 'llm',
			subjectTypes: ['missing_type'],
			objectTypes: ['document']
		};
		const report = evaluateSchemaQuality(schema([recordedIn]), {
			discoveryProposals: [{ entityTypes: [], relationTypes: [recordedIn] }],
			entityConsolidation: 'complete',
			relationConsolidation: 'complete',
			minimumScore: 0
		});

		assert.equal(report.status, 'failed');
		assert.ok(report.issues.some((issue) => issue.code === 'schema-not-closed'));
		assert.ok(report.issues.some((issue) => issue.code === 'non-semantic-relation'));
		assert.match(schemaQualityFailureMessage(report), /hard requirement failed/);
	});

	it('records fallback and overly broad endpoint sets as warnings', () => {
		const broadRelation: SchemaCategory = {
			name: 'supports_operation',
			description: 'An entity supports an operation',
			source: 'llm',
			subjectTypes: entityTypes.map((type) => type.name),
			objectTypes: ['process']
		};
		const report = evaluateSchemaQuality(schema([broadRelation]), {
			discoveryProposals: [{ entityTypes: [], relationTypes: [broadRelation] }],
			entityConsolidation: 'failed',
			relationConsolidation: 'chunked',
			minimumScore: 0.5
		});

		assert.equal(report.status, 'passed');
		assert.ok(report.issues.some((issue) => issue.code === 'broad-endpoints'));
		assert.ok(report.issues.some((issue) => issue.code === 'consolidation-fallback'));
	});
});
