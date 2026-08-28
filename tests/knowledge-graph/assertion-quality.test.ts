import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	hasConsistentCondition,
	hasConsistentModality,
	isSemanticAssertionCandidate,
	isSemanticRelationCategory,
	isDocumentLocator,
	isGroupTypedAsPerson,
	looksLikeClause,
	looksLikeTemplate
} from '../../src/lib/server/knowledge-graph-new/assertion-quality';
import type {
	CorpusSchema,
	ExtractedAssertion,
	SchemaCategory
} from '../../src/lib/server/knowledge-graph-new/extraction';

const relations: SchemaCategory[] = [
	{
		name: 'uses_equipment',
		description: 'A process uses equipment',
		source: 'test',
		subjectTypes: ['process'],
		objectTypes: ['object']
	},
	{
		name: 'is_detailed_in',
		description: 'A process is detailed in a document',
		source: 'test',
		subjectTypes: ['process'],
		objectTypes: ['document']
	},
	{
		name: 'governs_procedure',
		description: 'A document governs a process',
		source: 'test',
		subjectTypes: ['document'],
		objectTypes: ['process']
	},
	{
		name: 'is_related_to_document',
		description: 'A topic is related to a document',
		source: 'test',
		subjectTypes: ['process'],
		objectTypes: ['document']
	},
	{
		name: 'is_used_for_training',
		description: 'A role is used for training',
		source: 'test',
		subjectTypes: ['role', 'person_group'],
		objectTypes: ['process']
	}
];

const schema: CorpusSchema = {
	entityTypes: [
		{ name: 'process', description: 'A process', source: 'test' },
		{ name: 'object', description: 'An object', source: 'test' },
		{ name: 'document', description: 'A document', source: 'test' }
	],
	relationTypes: relations,
	sampledChunkIds: [],
	version: 'test'
};

function assertion(overrides: Partial<ExtractedAssertion> = {}): ExtractedAssertion {
	return {
		subject: 'airway procedure',
		subjectType: 'process',
		rawPredicate: 'uses_equipment',
		object: 'airway cannula',
		objectType: 'object',
		evidence: 'The airway procedure uses an airway cannula.',
		startDate: null,
		endDate: null,
		status: 'asserted',
		modality: 'observed',
		modalityCue: null,
		condition: null,
		extractors: ['llm'],
		verified: false,
		score: null,
		offsets: null,
		...overrides
	};
}

describe('semantic assertion quality', () => {
	it('keeps explicit semantic relationships', () => {
		const candidate = assertion();
		assert.equal(isSemanticAssertionCandidate(candidate.evidence, candidate, schema), true);
	});

	it('requires explicit modal language to retain its modality', () => {
		assert.equal(
			hasConsistentModality('Pilots must complete the evaluation.', 'required', 'must'),
			true
		);
		assert.equal(
			hasConsistentModality('Pilots must complete the evaluation.', 'observed', null),
			false
		);
		assert.equal(
			hasConsistentModality('Pilots should arrest the descent.', 'recommended', 'should'),
			true
		);
		assert.equal(
			hasConsistentModality('Pilots should arrest the descent.', 'habitual', null),
			false
		);
	});

	it('distinguishes permission from possibility and preserves exact cues', () => {
		assert.equal(
			hasConsistentModality('The commander may approve the request.', 'permitted', 'may'),
			true
		);
		assert.equal(
			hasConsistentModality('Turbulence may affect the approach.', 'observed', 'may', 'uncertain'),
			true
		);
		assert.equal(
			hasConsistentModality('Turbulence may affect the approach.', 'observed', 'may'),
			false
		);
		assert.equal(
			hasConsistentModality('The commander may approve the request.', 'permitted', 'MAY'),
			false
		);
	});

	it('keeps prohibitions asserted and conditions verbatim', () => {
		const evidence = 'Pilots must not descend below the minimum unless directed by ATC.';
		assert.equal(
			hasConsistentModality(
				evidence,
				'prohibited',
				'must not',
				'asserted',
				'unless directed by ATC'
			),
			true
		);
		assert.equal(
			hasConsistentModality(
				evidence,
				'prohibited',
				'must not',
				'negated',
				'unless directed by ATC'
			),
			false
		);
		assert.equal(hasConsistentCondition(evidence, null), false);
		assert.equal(hasConsistentCondition(evidence, 'unless directed by ATC'), true);
		assert.equal(hasConsistentCondition(evidence, 'if directed by ATC'), false);
	});

	it('classifies document containment as provenance', () => {
		assert.equal(isSemanticRelationCategory(relations[1]), false);
		assert.equal(
			isSemanticAssertionCandidate(
				'Cricothyroidotomy appears in the handbook.',
				assertion({
					rawPredicate: 'is_detailed_in',
					object: 'handbook',
					objectType: 'document',
					evidence: 'Cricothyroidotomy appears in the handbook.'
				}),
				schema
			),
			false
		);
	});

	it('classifies publication details-procedure relations as provenance', () => {
		assert.equal(
			isSemanticRelationCategory({
				name: 'details_procedure_for',
				description: 'A publication details a procedure',
				source: 'test',
				subjectTypes: ['military_publication'],
				objectTypes: ['process']
			}),
			false
		);
	});

	it('rejects vague related-to predicates even when schema discovery returns them', () => {
		assert.equal(isSemanticRelationCategory(relations[3]), false);
	});

	it('rejects passive-use relations that treat actors as the used object', () => {
		assert.equal(isSemanticRelationCategory(relations[4]), false);
	});

	it('rejects strong template passages and clause endpoints', () => {
		const template =
			'Opening Sentence. Captain Example distinguished herself by (achievement). Narrative Description. _____';
		assert.equal(looksLikeTemplate(template), true);
		assert.equal(isSemanticAssertionCandidate(template, assertion(), schema), false);
		assert.equal(
			looksLikeClause('waiver authority is designated using tier levels for each statement'),
			true
		);
	});

	it('rejects citation-only evidence presented as governance', () => {
		const candidate = assertion({
			subject: 'DAFI 35-101',
			subjectType: 'document',
			rawPredicate: 'governs_procedure',
			object: 'Public Affairs Operations',
			objectType: 'process',
			evidence: 'See DAFI 35-101, Public Affairs Operations for more information.'
		});

		assert.equal(isSemanticAssertionCandidate(candidate.evidence, candidate, schema), false);
	});

	it('keeps document locators in provenance rather than entity endpoints', () => {
		assert.equal(isDocumentLocator('paragraph 2.14.5', 'military_directive'), true);
		assert.equal(isDocumentLocator('Appendix E', 'document'), true);
		assert.equal(isDocumentLocator('Title 10 USC Section 772', 'military_regulation'), false);
		assert.equal(
			isSemanticAssertionCandidate(
				'Commanders follow paragraph 2.14.5.',
				assertion({
					subject: 'Commanders',
					subjectType: 'role',
					rawPredicate: 'governs_procedure',
					object: 'paragraph 2.14.5',
					objectType: 'military_directive',
					evidence: 'Commanders follow paragraph 2.14.5.'
				}),
				schema
			),
			false
		);
	});

	it('distinguishes groups and roles from named individuals', () => {
		assert.equal(isGroupTypedAsPerson('All pilots', 'person'), true);
		assert.equal(isGroupTypedAsPerson('Instructor pilots', 'person'), true);
		assert.equal(isGroupTypedAsPerson('IPs', 'person'), true);
		assert.equal(isGroupTypedAsPerson('Jerry T. Smythe', 'person'), false);
		assert.equal(isGroupTypedAsPerson('All pilots', 'person_group'), false);
	});
});
