import assert from 'node:assert/strict';
import test from 'node:test';
import { GraphStore } from './graph-store';
import { compilePprIndex } from './ppr-index';
import { graphId } from './utils';

test('PPR index compiles weighted edges in both directions', () => {
	const graph = new GraphStore();
	const chunk = graphId('chunk', 'chunk-1');
	const entity = graphId('entity', 'airway');
	graph.addNode({ id: chunk, label: 'Chunk 1', kind: 'chunk', chunkId: 'chunk-1' });
	graph.addNode({ id: entity, label: 'airway', kind: 'entity', entityKind: 'concept' });
	graph.addEdge({
		source: chunk,
		target: entity,
		relation: 'MENTIONS',
		weight: 2,
		evidence: '',
		chunkId: 'chunk-1'
	});

	const index = compilePprIndex(graph);
	assert.deepEqual(index.nodeIdByIndex, [chunk, entity]);
	assert.deepEqual([...index.offsets], [0, 1, 2]);
	assert.deepEqual([...index.targets], [1, 0]);
	assert.deepEqual([...index.weights], [2, 2]);
	assert.deepEqual([...index.outgoingWeights], [2, 2]);
});

test('PPR index excludes document containment hubs and noisy entities', () => {
	const graph = new GraphStore();
	const document = graphId('document', 'manual');
	const chunk = graphId('chunk', 'chunk-1');
	const noisyEntity = graphId('entity', 'the');

	graph.addNode({ id: document, label: 'Manual', kind: 'document', documentId: 'manual' });
	graph.addNode({ id: chunk, label: 'Chunk 1', kind: 'chunk', chunkId: 'chunk-1' });
	graph.addNode({ id: noisyEntity, label: 'the', kind: 'entity', entityKind: 'concept' });
	graph.addEdge({
		source: document,
		target: chunk,
		relation: 'CONTAINS',
		weight: 0.5,
		evidence: '',
		chunkId: 'chunk-1'
	});
	graph.addEdge({
		source: chunk,
		target: noisyEntity,
		relation: 'MENTIONS',
		weight: 1,
		evidence: '',
		chunkId: 'chunk-1'
	});

	const index = compilePprIndex(graph);
	assert.deepEqual(index.nodeIdByIndex, [chunk]);
	assert.deepEqual([...index.offsets], [0, 0]);
	assert.equal(index.targets.length, 0);
});
