export interface GraphSearchOptions {
	query: string;
	topK: number;

	// Hybrid retrieval can provide grounded starting chunks.
	seedChunkIds?: string[];

	usePpr: boolean;
	damping?: number;
}

export interface GraphSearchChunk {
	chunkId: string;
	score: number;
	supportingAssertionIds: string[];
}

export interface GraphSearchPath {
	// Keep these in traversal order. The benchmark uses the order to detect
	// reversed and otherwise incorrect graph paths.
	assertionIds: string[];
	score: number;
}

export interface GraphSearchResult {
	chunks: GraphSearchChunk[];
	paths: GraphSearchPath[];
}

export async function searchKnowledgeGraph(
	_options: GraphSearchOptions
): Promise<GraphSearchResult> {
	// Load the completed KnowledgeGraph and run graph traversal/PPR here.
	// Do not put benchmark data in this function.
	throw new Error('Graph search has not been implemented.');
}
