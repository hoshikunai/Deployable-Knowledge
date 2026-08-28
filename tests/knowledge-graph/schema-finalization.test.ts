import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	finalizeCorpusSchema,
	relationAcceptsTypes
} from '../../src/lib/server/knowledge-graph-new/schema-finalization';
import type { SchemaCategory } from '../../src/lib/server/knowledge-graph-new/extraction';

const universal: SchemaCategory[] = [
	{ name: 'document', description: 'A document', source: 'universal' },
	{ name: 'process', description: 'A process', source: 'universal' }
];

describe('adaptive corpus schema finalization', () => {
	it('promotes missing relation endpoint types and guarantees schema closure', () => {
		const schema = finalizeCorpusSchema({
			universalEntityTypes: universal,
			proposedEntityTypes: [
				{ name: 'string', description: 'A primitive string value', source: 'llm' }
			],
			proposedRelationTypes: [
				{
					name: 'governs_protocol',
					description: 'A document governs a protocol',
					source: 'llm',
					subjectTypes: ['document'],
					objectTypes: ['protocol']
				}
			],
			sampledChunkIds: ['chunk-1'],
			version: 'test',
			maxEntityTypes: 6,
			maxRelationTypes: 4
		});

		assert.ok(schema.entityTypes.some((type) => type.name === 'protocol'));
		assert.equal(schema.relationTypes.length, 1);
		assert.equal(
			schema.entityTypes.find((type) => type.name === 'protocol')?.source,
			'relation-endpoint'
		);
	});

	it('prioritizes relation endpoint types when the vocabulary is bounded', () => {
		const schema = finalizeCorpusSchema({
			universalEntityTypes: universal,
			proposedEntityTypes: [
				{ name: 'unused_type', description: 'Not used by a relation', source: 'llm' },
				{ name: 'medical_device', description: 'A medical device', source: 'llm' }
			],
			proposedRelationTypes: [
				{
					name: 'requires_equipment',
					description: 'A process requires equipment',
					source: 'llm',
					subjectTypes: ['process'],
					objectTypes: ['medical_device']
				}
			],
			sampledChunkIds: [],
			version: 'test',
			maxEntityTypes: 3,
			maxRelationTypes: 4
		});

		assert.deepEqual(
			schema.entityTypes.map((type) => type.name),
			['document', 'process', 'medical_device']
		);
	});

	it('does not allow unknown or other to bypass relation type constraints', () => {
		const relation: SchemaCategory = {
			name: 'requires_equipment',
			description: 'A process requires equipment',
			source: 'llm',
			subjectTypes: ['process'],
			objectTypes: ['medical_device']
		};

		assert.equal(relationAcceptsTypes(relation, 'process', 'medical_device'), true);
		assert.equal(relationAcceptsTypes(relation, 'unknown', 'medical_device'), false);
		assert.equal(relationAcceptsTypes(relation, 'process', 'other'), false);
		assert.equal(relationAcceptsTypes(relation, 'document', 'medical_device'), false);
	});

	it('removes provenance relations and merges semantic relation variants', () => {
		const schema = finalizeCorpusSchema({
			universalEntityTypes: universal,
			proposedEntityTypes: [],
			proposedRelationTypes: [
				{
					name: 'is_detailed_in',
					description: 'A process is detailed in a document',
					source: 'llm',
					subjectTypes: ['process'],
					objectTypes: ['document']
				},
				{
					name: 'references_document',
					description: 'A document references another document',
					source: 'llm',
					subjectTypes: ['document'],
					objectTypes: ['document']
				},
				{
					name: 'is_recorded_in',
					description: 'A process is recorded in a document',
					source: 'llm',
					subjectTypes: ['process'],
					objectTypes: ['document']
				},
				{
					name: 'contains_information_about',
					description: 'A document contains information about a string',
					source: 'llm',
					subjectTypes: ['document'],
					objectTypes: ['string']
				},
				{
					name: 'governs_procedure',
					description: 'A document governs a process',
					source: 'llm',
					subjectTypes: ['document'],
					objectTypes: ['process']
				},
				{
					name: 'governs_procedure_in',
					description: 'A document governs a procedure in the corpus',
					source: 'llm',
					subjectTypes: ['document'],
					objectTypes: ['process']
				}
			],
			sampledChunkIds: [],
			version: 'test',
			maxEntityTypes: 4,
			maxRelationTypes: 4
		});

		assert.deepEqual(
			schema.relationTypes.map((relation) => relation.name),
			['governs_procedure']
		);
		assert.equal(
			schema.entityTypes.some((type) => type.name === 'string'),
			false
		);
	});

	it('supports plural role holders and merges redundant document qualifiers', () => {
		const schema = finalizeCorpusSchema({
			universalEntityTypes: [
				...universal,
				{ name: 'role', description: 'A role', source: 'universal' },
				{ name: 'person_group', description: 'A group', source: 'universal' }
			],
			proposedEntityTypes: [],
			proposedRelationTypes: [
				{
					name: 'is_governed_by',
					description: 'A role is governed by a document',
					source: 'llm',
					subjectTypes: ['role'],
					objectTypes: ['document']
				},
				{
					name: 'is_governed_by_policy',
					description: 'A role is governed by a policy document',
					source: 'llm',
					subjectTypes: ['role'],
					objectTypes: ['document']
				}
			],
			sampledChunkIds: [],
			version: 'test',
			maxEntityTypes: 4,
			maxRelationTypes: 4
		});

		assert.equal(schema.relationTypes.length, 1);
		assert.deepEqual(schema.relationTypes[0].subjectTypes, ['role', 'person_group']);
		assert.equal(relationAcceptsTypes(schema.relationTypes[0], 'person_group', 'document'), true);
	});

	it('merges relation names that differ only by a leading auxiliary verb', () => {
		const schema = finalizeCorpusSchema({
			universalEntityTypes: universal,
			proposedEntityTypes: [],
			proposedRelationTypes: [
				{
					name: 'is_governed_by',
					description: 'A process is governed by a document',
					source: 'llm',
					subjectTypes: ['process'],
					objectTypes: ['document']
				},
				{
					name: 'governed_by',
					description: 'A process is governed by a document',
					source: 'llm',
					subjectTypes: ['process'],
					objectTypes: ['document']
				}
			],
			sampledChunkIds: [],
			version: 'test',
			maxEntityTypes: 2,
			maxRelationTypes: 4
		});

		assert.deepEqual(
			schema.relationTypes.map((relation) => relation.name),
			['is_governed_by']
		);
	});
});
