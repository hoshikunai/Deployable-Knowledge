import assert from 'node:assert/strict';
import test from 'node:test';
import { GraphStore } from './graph-store';
import { personalizedPageRank } from './personalized-page-rank';
import { compilePprIndex } from './ppr-index';
import type { GraphSeedCandidate } from './seed-selection';
import { graphId } from './utils';

test('PPR propagates entity relevance to indirectly connected chunks', () => {
	const graph = new GraphStore();
	const firstChunk = graphId('chunk', 'chunk-1');
	const secondChunk = graphId('chunk', 'chunk-2');
	const march = graphId('entity', 'MARCH');
	const hemorrhage = graphId('entity', 'massive hemorrhage');

	graph.addNode({
		id: firstChunk,
		label: 'MARCH overview',
		kind: 'chunk',
		chunkId: 'chunk-1'
	});
	graph.addNode({
		id: secondChunk,
		label: 'Hemorrhage treatment',
		kind: 'chunk',
		chunkId: 'chunk-2'
	});
	graph.addNode({
		id: march,
		label: 'MARCH',
		kind: 'entity',
		entityKind: 'protocol'
	});
	graph.addNode({
		id: hemorrhage,
		label: 'massive hemorrhage',
		kind: 'entity',
		entityKind: 'condition'
	});

	graph.addEdge({
		source: firstChunk,
		target: march,
		relation: 'MENTIONS',
		weight: 1,
		evidence: 'MARCH protocol',
		chunkId: 'chunk-1'
	});
	graph.addEdge({
		source: march,
		target: hemorrhage,
		relation: 'HAS_STEP',
		weight: 2,
		evidence: 'MARCH begins with massive hemorrhage',
		chunkId: 'chunk-1'
	});
	graph.addEdge({
		source: hemorrhage,
		target: secondChunk,
		relation: 'MENTIONS',
		weight: 1,
		evidence: 'Control massive hemorrhage',
		chunkId: 'chunk-2'
	});

	const seeds: GraphSeedCandidate[] = [
		{
			nodeId: march,
			kind: 'entity',
			label: 'MARCH',
			score: 1,
			sources: ['entity-exact'],
			sourceScores: { 'entity-exact': 1 }
		}
	];

	const results = personalizedPageRank(compilePprIndex(graph), seeds, {
		resultLimit: 10
	});

	const first = results.find((result) => result.chunkId === 'chunk-1');
	const second = results.find((result) => result.chunkId === 'chunk-2');

	assert.ok(first);
	assert.ok(second);
	assert.ok(first.score > second.score);
	assert.ok(second.score > 0);
});

test('PPR does not spread through document containment hubs', () => {
	const graph = new GraphStore();
	const document = graphId('document', 'manual');
	const relevantChunk = graphId('chunk', 'relevant');
	const unrelatedChunk = graphId('chunk', 'unrelated');

	graph.addNode({
		id: document,
		label: 'Manual',
		kind: 'document',
		documentId: 'manual'
	});
	graph.addNode({
		id: relevantChunk,
		label: 'Relevant',
		kind: 'chunk',
		chunkId: 'relevant'
	});
	graph.addNode({
		id: unrelatedChunk,
		label: 'Unrelated',
		kind: 'chunk',
		chunkId: 'unrelated'
	});

	for (const chunk of [relevantChunk, unrelatedChunk]) {
		graph.addEdge({
			source: document,
			target: chunk,
			relation: 'CONTAINS',
			weight: 0.5,
			evidence: '',
			chunkId: graph.getNode(chunk)?.chunkId
		});
	}

	const seeds: GraphSeedCandidate[] = [
		{
			nodeId: relevantChunk,
			kind: 'chunk',
			label: 'Relevant',
			chunkId: 'relevant',
			score: 1,
			sources: ['hybrid'],
			sourceScores: { hybrid: 1 }
		}
	];

	const results = personalizedPageRank(compilePprIndex(graph), seeds);

	assert.ok(results.some((result) => result.chunkId === 'relevant'));
	assert.ok(!results.some((result) => result.chunkId === 'unrelated'));
});

test('PPR returns no results without usable seeds', () => {
	assert.deepEqual(personalizedPageRank(compilePprIndex(new GraphStore()), []), []);
});

test('PPR enforces resultLimit without sorting every graph node', () => {
	const graph = new GraphStore();
	const entity = graphId('entity', 'shared concept');
	graph.addNode({ id: entity, label: 'shared concept', kind: 'entity', entityKind: 'concept' });

	for (const chunkId of ['third', 'first', 'second']) {
		const chunkNodeId = graphId('chunk', chunkId);
		graph.addNode({ id: chunkNodeId, label: chunkId, kind: 'chunk', chunkId });
		graph.addEdge({
			source: entity,
			target: chunkNodeId,
			relation: 'MENTIONS',
			weight: 1,
			evidence: '',
			chunkId
		});
	}

	const seeds: GraphSeedCandidate[] = [
		{
			nodeId: entity,
			kind: 'entity',
			label: 'shared concept',
			score: 1,
			sources: ['entity-exact'],
			sourceScores: { 'entity-exact': 1 }
		}
	];

	const results = personalizedPageRank(compilePprIndex(graph), seeds, { resultLimit: 2 });
	assert.deepEqual(
		results.map((result) => result.chunkId),
		['first', 'second']
	);
});
