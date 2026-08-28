import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
	buildSchemaDiscoveryBatches,
	buildSchemaSample
} from '../../src/lib/server/knowledge-graph-new/schema-sampling';

const chunks = Array.from({ length: 12 }, (_, index) => ({
	chunkId: `private-chunk-id-${index}`,
	documentId: `private-document-id-${Math.floor(index / 4)}`,
	content: `Corpus section ${index} describes a reusable domain concept and its relationships.`
}));

describe('schema discovery sampling', () => {
	it('uses compact anonymous labels instead of full database identifiers', () => {
		const sample = buildSchemaSample(chunks, { maxChunks: 6, maxCharacters: 1_200 });

		assert.ok(sample.text.includes('[document:d1 sample:1]'));
		assert.equal(sample.text.includes('private-document-id'), false);
		assert.equal(sample.text.includes('private-chunk-id'), false);
	});

	it('creates bounded corpus-diverse discovery batches', () => {
		const batches = buildSchemaDiscoveryBatches(chunks, {
			maxBatches: 3,
			maxCharactersPerBatch: 800
		});

		assert.equal(batches.length, 3);
		assert.ok(batches.every((batch) => batch.length <= 800));
		assert.ok(batches.every((batch) => batch.includes('Corpus section')));
		assert.ok(batches[0].includes('Corpus section 0'));
		assert.ok(batches[0].includes('Corpus section 3'));
	});
});
