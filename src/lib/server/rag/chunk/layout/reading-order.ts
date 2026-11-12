import type { LayoutCluster, LayoutLabel, PageSize } from './layout-types';

// Port of docling's rule-based reading order (docling/models/postprocessing/reading_order_rb.py,
// _predict_page) without separators, captions or merges. Boxes are flipped to a bottom-left
// origin so every predicate matches the original. Elements link left-to-right when they are
// consecutive and side by side, and top-to-bottom when one sits above the other with nothing
// between; a depth-first walk from the topmost heads, which climbs to unvisited ancestors
// before emitting each element, yields columns in order.

type Node = { cid: number; label: LayoutLabel; l: number; r: number; t: number; b: number };

type Graph = { up: number[][]; down: number[][] };

const EPSILON = 1e-3;
const VERTICAL_IOU_THRESHOLD = 0.8;
const QUERY_PADDING = 0.1;
const INTERRUPTION_PADDING = 1.0;
const NEAR_VERTICAL_OVERLAP_NORM = 0.0025;
const LEFT_EDGE_ALIGNMENT_NORM = 0.01;
const HORIZONTAL_DILATION_NORM = 0.15;
const GRAPHIC_LABELS = new Set<LayoutLabel>(['table', 'picture', 'code']);

function strictlyAbove(a: Node, b: Node): boolean {
	return a.b + EPSILON > b.t;
}

function strictlyLeftOf(a: Node, b: Node): boolean {
	return a.r + EPSILON < b.l;
}

function overlapsHorizontally(a: Node, b: Node): boolean {
	return !(a.r <= b.l || b.r <= a.l);
}

function overlapsVertically(a: Node, b: Node): boolean {
	return !(a.t <= b.b || b.t <= a.b);
}

function overlaps(a: Node, b: Node): boolean {
	return overlapsHorizontally(a, b) && overlapsVertically(a, b);
}

function verticalIouAbove(a: Node, b: Node, threshold: number): boolean {
	if (!overlapsVertically(a, b)) return false;
	const union = Math.max(a.t, b.t) - Math.min(a.b, b.b);
	const intersection = Math.min(a.t, b.t) - Math.max(a.b, b.b);
	return union > 0 && intersection / union > threshold;
}

function precedes(a: Node, b: Node): boolean {
	if (overlapsHorizontally(a, b)) return a.b > b.b;
	return a.l < b.l;
}

function compareNodes(a: Node, b: Node): number {
	if (precedes(a, b)) return -1;
	if (precedes(b, a)) return 1;
	return 0;
}

function hasElementBetween(nodes: Node[], left: number, right: number): boolean {
	const a = nodes[left];
	const b = nodes[right];
	const bottom = Math.max(a.b, b.b);
	const top = Math.min(a.t, b.t);
	return nodes.some(
		(node, index) =>
			index !== left &&
			index !== right &&
			a.r < node.r &&
			node.l < b.l &&
			bottom < node.t &&
			node.b < top
	);
}

function leftToRightLinks(nodes: Node[]): Map<number, number> {
	const links = new Map<number, number>();
	nodes.forEach((a, i) => {
		nodes.forEach((b, j) => {
			if (
				a.cid + 1 === b.cid &&
				strictlyLeftOf(a, b) &&
				verticalIouAbove(a, b, VERTICAL_IOU_THRESHOLD) &&
				!hasElementBetween(nodes, i, j)
			) {
				links.set(i, j);
			}
		});
	});
	return links;
}

function hasInterruption(nodes: Node[], i: number, j: number): boolean {
	const upper = nodes[i];
	const lower = nodes[j];
	const xMin = Math.min(upper.l, lower.l) - INTERRUPTION_PADDING;
	const xMax = Math.max(upper.r, lower.r) + INTERRUPTION_PADDING;
	const yMin = Math.min(lower.t, upper.b);
	const yMax = Math.max(lower.t, upper.b);

	return nodes.some((node, w) => {
		if (w === i || w === j) return false;
		if (node.r < xMin || node.l > xMax || node.t < yMin || node.b > yMax) return false;
		const overlapsSequence = overlapsHorizontally(upper, node) || overlapsHorizontally(lower, node);
		return overlapsSequence && strictlyAbove(upper, node) && strictlyAbove(node, lower);
	});
}

function addEdge(graph: Graph, upper: number, lower: number): void {
	if (!graph.down[upper].includes(lower)) graph.down[upper].push(lower);
	if (!graph.up[lower].includes(upper)) graph.up[lower].push(upper);
}

function buildGraph(nodes: Node[], links: Map<number, number>, page: PageSize): Graph {
	const graph: Graph = { up: nodes.map(() => []), down: nodes.map(() => []) };
	const rightToLeft = new Map([...links].map(([left, right]) => [right, left]));

	nodes.forEach((lower, j) => {
		const partner = rightToLeft.get(j);
		if (partner !== undefined) addEdge(graph, partner, j);

		nodes.forEach((upper, candidate) => {
			if (candidate === j) return;
			if (upper.r < lower.l - QUERY_PADDING || upper.l > lower.r + QUERY_PADDING) return;
			if (upper.t < lower.t) return;
			if (!strictlyAbove(upper, lower) || !overlapsHorizontally(upper, lower)) return;
			if (hasInterruption(nodes, candidate, j)) return;

			let i = candidate;
			while (links.has(i)) i = links.get(i) as number;
			graph.down[i].push(j);
			graph.up[j].push(i);
		});
	});

	const byCid = new Map(nodes.map((node, index) => [node.cid, index]));
	nodes.forEach((upper, i) => {
		if (GRAPHIC_LABELS.has(upper.label)) return;
		const j = byCid.get(upper.cid + 1);
		if (j === undefined) return;
		const lower = nodes[j];
		if (GRAPHIC_LABELS.has(lower.label)) return;
		const tolerance = page.height * NEAR_VERTICAL_OVERLAP_NORM;
		if (
			Math.abs(upper.l - lower.l) < page.width * LEFT_EDGE_ALIGNMENT_NORM &&
			upper.b < lower.t &&
			lower.t <= upper.b + tolerance &&
			upper.t > lower.t &&
			upper.b > lower.b &&
			!graph.down[i].includes(j)
		) {
			graph.down[i].push(j);
			graph.up[j].push(i);
		}
	});

	return graph;
}

function isOneToOne(graph: Graph, upper: number, lower: number): boolean {
	return (
		graph.down[upper].length === 1 &&
		graph.down[upper][0] === lower &&
		graph.up[lower].length === 1 &&
		graph.up[lower][0] === upper
	);
}

function widen(
	range: [number, number],
	neighbour: Node,
	threshold: number
): [number, number] | null {
	const left = Math.min(range[0], neighbour.l);
	const right = Math.max(range[1], neighbour.r);
	if (range[0] - left > threshold || right - range[1] > threshold) return null;
	return [left, right];
}

function dilate(nodes: Node[], graph: Graph, page: PageSize): Node[] {
	const threshold = HORIZONTAL_DILATION_NORM * page.width;
	return nodes.map((node, i) => {
		let range: [number, number] | null = [node.l, node.r];

		const up = graph.up[i][0];
		if (up !== undefined && isOneToOne(graph, up, i)) {
			range = widen(range, nodes[up], threshold);
			if (!range) return node;
		}

		const down = graph.down[i][0];
		if (down !== undefined && isOneToOne(graph, i, down)) {
			range = widen(range, nodes[down], threshold);
			if (!range) return node;
		}

		const dilated = { ...node, l: range[0], r: range[1] };
		const collides = nodes.some((other, j) => j !== i && overlaps(other, dilated));
		return collides ? node : dilated;
	});
}

function separatedAbove(nodes: Node[], upper: Node, lower: Node): boolean {
	if (overlapsHorizontally(upper, lower) || !strictlyAbove(upper, lower)) return false;
	return nodes.some(
		(node) =>
			node.cid !== upper.cid &&
			node.cid !== lower.cid &&
			strictlyAbove(upper, node) &&
			strictlyAbove(node, lower) &&
			overlapsHorizontally(node, upper) &&
			overlapsHorizontally(node, lower)
	);
}

function findHeads(nodes: Node[], graph: Graph): number[] {
	const heads = nodes.map((_, index) => index).filter((index) => graph.up[index].length === 0);
	return heads.sort((a, b) => {
		if (separatedAbove(nodes, nodes[a], nodes[b])) return -1;
		if (separatedAbove(nodes, nodes[b], nodes[a])) return 1;
		return precedes(nodes[a], nodes[b]) ? -1 : 1;
	});
}

function climbToUnvisited(start: number, visited: boolean[], graph: Graph): number {
	let current = start;
	const walked = new Set([start]);
	for (;;) {
		const next = graph.up[current].find((index) => !visited[index] && !walked.has(index));
		if (next === undefined) return current;
		walked.add(next);
		current = next;
	}
}

function walkDown(head: number, order: number[], visited: boolean[], graph: Graph): void {
	const stack: [number[], number][] = [[graph.down[head], 0]];
	while (stack.length > 0) {
		const [neighbours, offset] = stack[stack.length - 1];
		let advanced = false;
		for (let step = 0; offset + step < neighbours.length; step += 1) {
			const target = climbToUnvisited(neighbours[offset + step], visited, graph);
			if (visited[target]) continue;
			order.push(target);
			visited[target] = true;
			// docling stores the offset relative to the slice it iterated, not the full list
			stack[stack.length - 1] = [neighbours, step + 1];
			stack.push([graph.down[target], 0]);
			advanced = true;
			break;
		}
		if (!advanced) stack.pop();
	}
}

function toNode(cluster: LayoutCluster, cid: number, page: PageSize): Node {
	return {
		cid,
		label: cluster.label,
		l: cluster.box.l,
		r: cluster.box.r,
		t: page.height - cluster.box.t,
		b: page.height - cluster.box.b
	};
}

export function orderClusters(clusters: LayoutCluster[], page: PageSize): LayoutCluster[] {
	if (clusters.length < 2) return clusters;

	const nodes = clusters.map((cluster, cid) => toNode(cluster, cid, page));
	const links = leftToRightLinks(nodes);
	const initial = buildGraph(nodes, links, page);
	const graph = buildGraph(dilate(nodes, initial, page), links, page);

	const heads = findHeads(nodes, graph);
	for (const list of [...graph.up, ...graph.down]) {
		list.sort((a, b) => compareNodes(nodes[a], nodes[b]));
	}

	const visited = nodes.map(() => false);
	const order: number[] = [];
	for (const head of heads) {
		if (visited[head]) continue;
		order.push(head);
		visited[head] = true;
		walkDown(head, order, visited, graph);
	}
	nodes.forEach((_, index) => {
		if (!visited[index]) order.push(index);
	});

	return order.map((index) => clusters[index]);
}
