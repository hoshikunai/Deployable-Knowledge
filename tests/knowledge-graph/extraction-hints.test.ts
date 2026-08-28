import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	buildExtractionOutput,
	hasExactMention,
	isGroundedAssertion,
	selectEntityHints,
	type CorpusSchema,
	type ExtractedEntity
} from '../../src/lib/server/knowledge-graph-new/extraction';

const schema: CorpusSchema = {
	entityTypes: [
		{ name: 'process', description: 'A process', source: 'test' },
		{ name: 'object', description: 'An object', source: 'test' }
	],
	relationTypes: [],
	sampledChunkIds: [],
	version: 'test'
};

describe('GLiNER entity hints', () => {
	it('keeps exact, schema-compatible spans and removes duplicate hints', () => {
		const text = 'The airway procedure uses an airway cannula.';
		const entities: ExtractedEntity[] = [
			{ text: 'airway procedure', type: 'process', start: 4, end: 20, score: 0.8 },
			{ text: 'airway procedure', type: 'process', start: 4, end: 20, score: 0.9 },
			{ text: 'airway cannula', type: 'object', start: 29, end: 43, score: 0.7 },
			{ text: 'airway', type: 'other', start: 4, end: 10, score: 0.99 },
			{ text: 'wrong span', type: 'object', start: 0, end: 10, score: 1 }
		];

		assert.deepEqual(selectEntityHints(text, schema, entities), [
			{ text: 'airway procedure', type: 'process', start: 4, end: 20, score: 0.9 },
			{ text: 'airway cannula', type: 'object', start: 29, end: 43, score: 0.7 }
		]);
	});

	it('bounds the number of hints sent to the model', () => {
		const text = 'one two three';
		const entities: ExtractedEntity[] = [
			{ text: 'one', type: 'object', start: 0, end: 3, score: 0.6 },
			{ text: 'two', type: 'object', start: 4, end: 7, score: 0.8 },
			{ text: 'three', type: 'object', start: 8, end: 13, score: 0.7 }
		];

		assert.deepEqual(
			selectEntityHints(text, schema, entities, 2).map((entity) => entity.text),
			['two', 'three']
		);
	});
});

describe('relation-dependent extraction schema', () => {
	it('binds each predicate to its declared endpoint types', () => {
		const extractionSchema = buildExtractionOutput({
			...schema,
			relationTypes: [
				{
					name: 'uses_equipment',
					description: 'A process uses equipment',
					source: 'test',
					subjectTypes: ['process'],
					objectTypes: ['object']
				},
				{
					name: 'governs',
					description: 'A document governs a process',
					source: 'test',
					subjectTypes: ['document'],
					objectTypes: ['process']
				}
			]
		}) as unknown as {
			properties: { assertions: { items: { oneOf: Array<Record<string, unknown>> } } };
		};
		const branches = extractionSchema.properties.assertions.items.oneOf as Array<{
			properties: Record<string, { const?: string; enum?: string[] }>;
		}>;

		assert.equal(branches.length, 2);
		assert.equal(branches[0].properties.rawPredicate.const, 'uses_equipment');
		assert.deepEqual(branches[0].properties.subjectType.enum, ['process']);
		assert.deepEqual(branches[0].properties.objectType.enum, ['object']);
		assert.deepEqual(branches[0].properties.modalityCue, { type: ['string', 'null'] });
		assert.deepEqual(branches[0].properties.condition, { type: ['string', 'null'] });
		assert.equal(branches[1].properties.rawPredicate.const, 'governs');
		assert.deepEqual(branches[1].properties.subjectType.enum, ['document']);
	});
});

describe('exact assertion grounding', () => {
	it('requires endpoint token boundaries instead of substrings', () => {
		assert.equal(hasExactMention('personal priorities', 'person'), false);
		assert.equal(hasExactMention('A person has personal priorities.', 'person'), true);
		assert.equal(hasExactMention('C2 agencies.', 'C2 agencies'), true);
		assert.equal(hasExactMention('DAFI 35-101, Public Affairs', 'DAFI 35-101'), true);
		assert.equal(
			isGroundedAssertion('Maintain personal priorities.', {
				subject: 'person',
				object: 'personal priorities',
				evidence: 'Maintain personal priorities.'
			}),
			false
		);
		assert.equal(
			isGroundedAssertion('A person maintains personal priorities.', {
				subject: 'person',
				object: 'personal priorities',
				evidence: 'A person maintains personal priorities.'
			}),
			true
		);
	});
});
