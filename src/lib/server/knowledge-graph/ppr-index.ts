import { GraphStore } from './graph-store';
import type { GraphEdge, GraphNode } from './types';
import { isNoisyEntityLabel } from './utils';

export type PprIndex = {
	nodeIndexById: Map<string, number>;
	nodeIdByIndex: string[];
	chunkIdByIndex: Array<string | null>;
	offsets: Uint32Array;
	targets: Uint32Array;
	weights: Float64Array;
	outgoingWeights: Float64Array;
};

export type CompilePprIndexOptions = {
	includeContainment?: boolean;
};

const MAX_UINT32 = 0xffffffff;

/**
 * Compile the object graph into an immutable compressed-sparse-row index.
 *
 * The two edge passes are intentional: the first sizes the typed arrays and the
 * second fills them, avoiding a temporary JavaScript object for every edge.
 */
export function compilePprIndex(graph: GraphStore, options: CompilePprIndexOptions = {}): PprIndex {
	const includeContainment = options.includeContainment === true;
	const nodeIndexById = new Map<string, number>();
	const nodeIdByIndex: string[] = [];
	const chunkIdByIndex: Array<string | null> = [];

	for (const node of graph.nodes.values()) {
		if (!isTraversableNode(node)) continue;

		const index = nodeIdByIndex.length;
		nodeIndexById.set(node.id, index);
		nodeIdByIndex.push(node.id);
		chunkIdByIndex.push(node.kind === 'chunk' ? (node.chunkId ?? null) : null);
	}

	const degrees = new Uint32Array(nodeIdByIndex.length);
	for (const edge of graph.edges) {
		if (!isIncludedEdge(edge, includeContainment)) continue;

		const source = nodeIndexById.get(edge.source);
		const target = nodeIndexById.get(edge.target);
		if (source === undefined || target === undefined) continue;

		degrees[source] += 1;
		degrees[target] += 1;
	}

	const offsets = new Uint32Array(nodeIdByIndex.length + 1);
	for (let index = 0; index < nodeIdByIndex.length; index += 1) {
		const nextOffset = offsets[index] + degrees[index];
		if (nextOffset > MAX_UINT32) {
			throw new Error('The PPR graph exceeds the Uint32 adjacency limit.');
		}
		offsets[index + 1] = nextOffset;
	}

	const targets = new Uint32Array(offsets[nodeIdByIndex.length]);
	const weights = new Float64Array(targets.length);
	const outgoingWeights = new Float64Array(nodeIdByIndex.length);
	const cursor = Uint32Array.from(offsets.subarray(0, nodeIdByIndex.length));

	for (const edge of graph.edges) {
		if (!isIncludedEdge(edge, includeContainment)) continue;

		const source = nodeIndexById.get(edge.source);
		const target = nodeIndexById.get(edge.target);
		if (source === undefined || target === undefined) continue;

		const weight = Number.isFinite(edge.weight) && edge.weight > 0 ? edge.weight : 1;
		addArc(source, target, weight, cursor, targets, weights, outgoingWeights);
		addArc(target, source, weight, cursor, targets, weights, outgoingWeights);
	}

	return {
		nodeIndexById,
		nodeIdByIndex,
		chunkIdByIndex,
		offsets,
		targets,
		weights,
		outgoingWeights
	};
}

function isIncludedEdge(edge: GraphEdge, includeContainment: boolean): boolean {
	return includeContainment || edge.relation !== 'CONTAINS';
}

function addArc(
	source: number,
	target: number,
	weight: number,
	cursor: Uint32Array,
	targets: Uint32Array,
	weights: Float64Array,
	outgoingWeights: Float64Array
): void {
	const index = cursor[source];
	cursor[source] += 1;
	targets[index] = target;
	weights[index] = weight;
	outgoingWeights[source] += weight;
}

function isTraversableNode(node: GraphNode): boolean {
	if (node.kind === 'document') return false;
	return node.kind !== 'entity' || !isNoisyEntityLabel(node.label, node.entityKind);
}
