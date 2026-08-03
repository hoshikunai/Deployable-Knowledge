import { GraphStore } from './graph-store';
import type { GraphEdge, GraphNode } from './types';
import type { GraphSeedCandidate } from './seed-selection';
import { isNoisyEntityLabel } from './utils';

export type PersonalizedPageRankOptions = {
	damping?: number;
	maxIterations?: number;
	tolerance?: number;
	resultLimit?: number;
	includeContainment?: boolean;
};

export type PersonalizedPageRankEvidence = {
	chunkId: string;
	score: number;
};

type CompiledGraph = {
	nodes: GraphNode[];
	nodeIndex: Map<string, number>;
	offsets: Uint32Array;
	targets: Uint32Array;
	weights: Float64Array;
	outgoingWeights: Float64Array;
};

const DEFAULT_DAMPING = 0.5;
const DEFAULT_MAX_ITERATIONS = 50;
const DEFAULT_TOLERANCE = 1e-7;
const DEFAULT_RESULT_LIMIT = 100;

export function personalizedPageRank(
	graph: GraphStore,
	seeds: readonly GraphSeedCandidate[],
	options: PersonalizedPageRankOptions = {}
): PersonalizedPageRankEvidence[] {
	if (!seeds.length || graph.nodes.size === 0) return [];

	const damping = clamp(options.damping ?? DEFAULT_DAMPING, 0, 0.999);
	const maxIterations = Math.max(
		1,
		Math.floor(options.maxIterations ?? DEFAULT_MAX_ITERATIONS)
	);
	const tolerance = Math.max(Number.EPSILON, options.tolerance ?? DEFAULT_TOLERANCE);
	const resultLimit = Math.max(
		0,
		Math.floor(options.resultLimit ?? DEFAULT_RESULT_LIMIT)
	);

	if (resultLimit === 0) return [];

	const compiled = compileGraph(graph, options.includeContainment === true);
	const reset = buildResetVector(compiled, seeds);

	if (!reset) return [];

	let current = reset.slice();
	let next = new Float64Array(compiled.nodes.length);

	for (let iteration = 0; iteration < maxIterations; iteration += 1) {
		next.fill(0);
		let danglingMass = 0;

		for (let nodeIndex = 0; nodeIndex < current.length; nodeIndex += 1) {
			const nodeScore = current[nodeIndex];
			if (nodeScore === 0) continue;

			const totalWeight = compiled.outgoingWeights[nodeIndex];
			if (totalWeight <= 0) {
				danglingMass += nodeScore;
				continue;
			}

			const start = compiled.offsets[nodeIndex];
			const end = compiled.offsets[nodeIndex + 1];
			const distributedScore = damping * nodeScore;

			for (let edgeIndex = start; edgeIndex < end; edgeIndex += 1) {
				const target = compiled.targets[edgeIndex];
				const weight = compiled.weights[edgeIndex];

				next[target] += distributedScore * (weight / totalWeight);
			}
		}

		const resetMass = 1 - damping + damping * danglingMass;
		let delta = 0;

		for (let nodeIndex = 0; nodeIndex < next.length; nodeIndex += 1) {
			next[nodeIndex] += resetMass * reset[nodeIndex];
			delta += Math.abs(next[nodeIndex] - current[nodeIndex]);
		}

		[current, next] = [next, current];

		if (delta <= tolerance) break;
	}

	return compiled.nodes
		.flatMap((node, index) => {
			const score = current[index];

			if (node.kind !== 'chunk' || !node.chunkId || score <= 0) return [];
			return [{ chunkId: node.chunkId, score }];
		})
		.sort((left, right) => right.score - left.score)
		.slice(0, resultLimit);
}

function buildResetVector(
	graph: CompiledGraph,
	seeds: readonly GraphSeedCandidate[]
): Float64Array | null {
	const reset = new Float64Array(graph.nodes.length);
	let totalWeight = 0;

	for (const seed of seeds) {
		const index = graph.nodeIndex.get(seed.nodeId);
		if (index === undefined) continue;

		const weight = Number.isFinite(seed.score) ? Math.max(0, seed.score) : 0;
		if (weight === 0) continue;

		reset[index] += weight;
		totalWeight += weight;
	}

	if (totalWeight === 0) return null;

	for (let index = 0; index < reset.length; index += 1) {
		reset[index] /= totalWeight;
	}

	return reset;
}

function compileGraph(graph: GraphStore, includeContainment: boolean): CompiledGraph {
	const nodes = [...graph.nodes.values()];
	const nodeIndex = new Map(nodes.map((node, index) => [node.id, index]));
	const traversable = nodes.map(isTraversableNode);
	const degrees = new Uint32Array(nodes.length);

	const includedEdges: Array<{
		source: number;
		target: number;
		weight: number;
	}> = [];

	for (const edge of graph.edges) {
		const compiledEdge = compileEdge(
			edge,
			nodeIndex,
			traversable,
			includeContainment
		);
		if (!compiledEdge) continue;

		includedEdges.push(compiledEdge);
		degrees[compiledEdge.source] += 1;
		degrees[compiledEdge.target] += 1;
	}

	const offsets = new Uint32Array(nodes.length + 1);
	for (let index = 0; index < nodes.length; index += 1) {
		offsets[index + 1] = offsets[index] + degrees[index];
	}

	const targets = new Uint32Array(offsets[nodes.length]);
	const weights = new Float64Array(offsets[nodes.length]);
	const outgoingWeights = new Float64Array(nodes.length);
	const cursor = Uint32Array.from(offsets.subarray(0, nodes.length));

	for (const edge of includedEdges) {
		addCompiledEdge(
			edge.source,
			edge.target,
			edge.weight,
			cursor,
			targets,
			weights,
			outgoingWeights
		);
		addCompiledEdge(
			edge.target,
			edge.source,
			edge.weight,
			cursor,
			targets,
			weights,
			outgoingWeights
		);
	}

	return {
		nodes,
		nodeIndex,
		offsets,
		targets,
		weights,
		outgoingWeights
	};
}

function compileEdge(
	edge: GraphEdge,
	nodeIndex: Map<string, number>,
	traversable: readonly boolean[],
	includeContainment: boolean
): { source: number; target: number; weight: number } | null {
	if (!includeContainment && edge.relation === 'CONTAINS') return null;

	const source = nodeIndex.get(edge.source);
	const target = nodeIndex.get(edge.target);

	if (source === undefined || target === undefined) return null;
	if (!traversable[source] || !traversable[target]) return null;

	const weight =
		Number.isFinite(edge.weight) && edge.weight > 0 ? edge.weight : 1;

	return { source, target, weight };
}

function addCompiledEdge(
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

	return (
		node.kind !== 'entity' ||
		!isNoisyEntityLabel(node.label, node.entityKind)
	);
}

function clamp(value: number, minimum: number, maximum: number): number {
	if (!Number.isFinite(value)) return minimum;
	return Math.max(minimum, Math.min(maximum, value));
}