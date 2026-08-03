import assert from 'node:assert/strict';
import test from 'node:test';
import { acronymDefinitionBoost, fuseKnowledgeGraphScores } from './knowledge-graph-search';

test('acronym definition evidence outranks incidental acronym text', () => {
	const query = 'What does MARCH stand for?';

	assert.equal(
		acronymDefinitionBoost(
			query,
			'MARCH (massive hemorrhage, airway, respirations, circulation, head injury/hypothermia)'
		),
		1
	);
	assert.equal(
		acronymDefinitionBoost(query, 'Roadmap updated in March (U) for the program office.'),
		0
	);
});

test('PPR contributes independently to the final retrieval score', () => {
	assert.equal(fuseKnowledgeGraphScores({ hybrid: 0, light: 0, path: 0, ppr: 1 }), 0.25);
	assert.equal(fuseKnowledgeGraphScores({ hybrid: 1, light: 1, path: 1, ppr: 1 }), 1);
});
