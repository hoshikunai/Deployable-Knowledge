import type { PprIndex } from './ppr-index';
import type { GraphSeedCandidate } from './seed-selection';

export type PersonalizedPageRankOptions = {
	damping?: number;
	maxIterations?: number;
	tolerance?: number;
	resultLimit?: number;
};

export type PersonalizedPageRankEvidence = {
	chunkId: string;
	score: number;
};

type ResetSeed = {
	index: number;
	weight: number;
};

const DEFAULT_DAMPING = 0.5;
const DEFAULT_MAX_ITERATIONS = 50;
const DEFAULT_TOLERANCE = 1e-7;
const DEFAULT_RESULT_LIMIT = 100;

export function personalizedPageRank(
	index: PprIndex,
	seeds: readonly GraphSeedCandidate[],
	options: PersonalizedPageRankOptions = {}
): PersonalizedPageRankEvidence[] {
	if (!seeds.length || index.nodeIdByIndex.length === 0) return [];

	const damping = clamp(options.damping ?? DEFAULT_DAMPING, 0, 0.999);
	const maxIterations = Math.max(1, Math.floor(options.maxIterations ?? DEFAULT_MAX_ITERATIONS));
	const tolerance = Math.max(Number.EPSILON, options.tolerance ?? DEFAULT_TOLERANCE);
	const resultLimit = Math.max(0, Math.floor(options.resultLimit ?? DEFAULT_RESULT_LIMIT));
	if (resultLimit === 0) return [];

	const resetSeeds = buildResetSeeds(index, seeds);
	if (!resetSeeds) return [];

	let current = new Float64Array(index.nodeIdByIndex.length);
	let next = new Float64Array(index.nodeIdByIndex.length);
	for (const seed of resetSeeds) current[seed.index] = seed.weight;

	for (let iteration = 0; iteration < maxIterations; iteration += 1) {
		next.fill(0);
		let danglingMass = 0;

		for (let nodeIndex = 0; nodeIndex < current.length; nodeIndex += 1) {
			const nodeScore = current[nodeIndex];
			if (nodeScore === 0) continue;

			const totalWeight = index.outgoingWeights[nodeIndex];
			if (totalWeight <= 0) {
				danglingMass += nodeScore;
				continue;
			}

			const start = index.offsets[nodeIndex];
			const end = index.offsets[nodeIndex + 1];
			const distributedScore = damping * nodeScore;

			for (let edgeIndex = start; edgeIndex < end; edgeIndex += 1) {
				const target = index.targets[edgeIndex];
				const weight = index.weights[edgeIndex];
				next[target] += distributedScore * (weight / totalWeight);
			}
		}

		const resetMass = 1 - damping + damping * danglingMass;
		for (const seed of resetSeeds) next[seed.index] += resetMass * seed.weight;

		let delta = 0;
		for (let nodeIndex = 0; nodeIndex < next.length; nodeIndex += 1) {
			delta += Math.abs(next[nodeIndex] - current[nodeIndex]);
		}

		[current, next] = [next, current];
		if (delta <= tolerance) break;
	}

	return selectTopChunkScores(index, current, resultLimit);
}

function buildResetSeeds(
	index: PprIndex,
	seeds: readonly GraphSeedCandidate[]
): ResetSeed[] | null {
	const weightsByIndex = new Map<number, number>();
	let totalWeight = 0;

	for (const seed of seeds) {
		const nodeIndex = index.nodeIndexById.get(seed.nodeId);
		if (nodeIndex === undefined) continue;

		const weight = Number.isFinite(seed.score) ? Math.max(0, seed.score) : 0;
		if (weight === 0) continue;

		weightsByIndex.set(nodeIndex, (weightsByIndex.get(nodeIndex) ?? 0) + weight);
		totalWeight += weight;
	}

	if (totalWeight === 0) return null;

	return [...weightsByIndex].map(([nodeIndex, weight]) => ({
		index: nodeIndex,
		weight: weight / totalWeight
	}));
}

function selectTopChunkScores(
	index: PprIndex,
	scores: Float64Array,
	limit: number
): PersonalizedPageRankEvidence[] {
	const heap: PersonalizedPageRankEvidence[] = [];

	for (let nodeIndex = 0; nodeIndex < scores.length; nodeIndex += 1) {
		const chunkId = index.chunkIdByIndex[nodeIndex];
		const score = scores[nodeIndex];
		if (!chunkId || score <= 0 || !Number.isFinite(score)) continue;

		const candidate = { chunkId, score };
		if (heap.length < limit) {
			heap.push(candidate);
			siftUpWorst(heap, heap.length - 1);
			continue;
		}

		if (!isBetter(candidate, heap[0])) continue;
		heap[0] = candidate;
		siftDownWorst(heap, 0);
	}

	return heap.sort(compareBestFirst);
}

function siftUpWorst(heap: PersonalizedPageRankEvidence[], start: number): void {
	let index = start;
	while (index > 0) {
		const parent = Math.floor((index - 1) / 2);
		if (!isWorse(heap[index], heap[parent])) return;
		[heap[index], heap[parent]] = [heap[parent], heap[index]];
		index = parent;
	}
}

function siftDownWorst(heap: PersonalizedPageRankEvidence[], start: number): void {
	let index = start;
	while (true) {
		const left = index * 2 + 1;
		const right = left + 1;
		let worst = index;

		if (left < heap.length && isWorse(heap[left], heap[worst])) worst = left;
		if (right < heap.length && isWorse(heap[right], heap[worst])) worst = right;
		if (worst === index) return;

		[heap[index], heap[worst]] = [heap[worst], heap[index]];
		index = worst;
	}
}

function isBetter(
	left: PersonalizedPageRankEvidence,
	right: PersonalizedPageRankEvidence
): boolean {
	return left.score > right.score || (left.score === right.score && left.chunkId < right.chunkId);
}

function isWorse(left: PersonalizedPageRankEvidence, right: PersonalizedPageRankEvidence): boolean {
	return left.score < right.score || (left.score === right.score && left.chunkId > right.chunkId);
}

function compareBestFirst(
	left: PersonalizedPageRankEvidence,
	right: PersonalizedPageRankEvidence
): number {
	return right.score - left.score || left.chunkId.localeCompare(right.chunkId);
}

function clamp(value: number, minimum: number, maximum: number): number {
	if (!Number.isFinite(value)) return minimum;
	return Math.max(minimum, Math.min(maximum, value));
}
