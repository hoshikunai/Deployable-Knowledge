import type { AssertionModality } from '../../src/lib/server/knowledge-graph-new/extraction';

export interface GoldEntity {
	canonical: string;
	aliases?: string[];
	type: string;
}

export interface GoldAssertion {
	id: string;
	documentId: string;
	chunkId: string;
	subject: GoldEntity;
	predicate: string;
	object: GoldEntity;
	evidence: string;
	status: 'asserted' | 'negated' | 'uncertain';
	modality: AssertionModality;
	// Required assertions count toward recall. Optional assertions count as
	// correct when produced but do not penalize the extractor when absent.
	required: boolean;
}

export interface GoldQuery {
	id: string;
	question: string;

	// Retrieval answer key
	relevantChunkIds: string[];

	// Paths are expressed using stable gold assertion IDs.
	expectedPathAssertionIds: string[][];

	// Useful for detecting confidently wrong retrieval.
	forbiddenChunkIds?: string[];
}

export interface GoldChunk {
	chunkId: string;
	documentId: string;

	// Detects corpus changes without storing source text in repo
	contentSha256: string;
}

export interface GoldBenchmark {
	version: string;
	corpusId: string;
	canonicalRelations: string[];
	chunks: GoldChunk[];
	assertions: GoldAssertion[];
	queries: GoldQuery[];
}
