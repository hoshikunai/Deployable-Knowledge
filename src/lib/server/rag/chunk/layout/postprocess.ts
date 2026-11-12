import {
	boxArea,
	intersectionOverSelf,
	intersectionOverUnion,
	unionBox,
	verticalOverlap
} from './geometry';
import type {
	LayoutCluster,
	LayoutDetection,
	LayoutLabel,
	PageSize,
	TextItemBox
} from './layout-types';

// Reduced port of docling's LayoutPostprocessor (docling/utils/layout_postprocessor.py):
// threshold detections, give every text run to the box covering most of it, turn leftover
// runs into text boxes, merge overlapping boxes, and nest regular boxes inside tables and
// pictures. Form and key-value containers are dropped; their contents stay regular boxes.

type OverlapParams = { areaThreshold: number; confThreshold: number };

const CONFIDENCE_THRESHOLDS: Record<LayoutLabel, number> = {
	caption: 0.5,
	footnote: 0.5,
	formula: 0.5,
	list_item: 0.5,
	page_footer: 0.5,
	page_header: 0.5,
	picture: 0.5,
	section_header: 0.45,
	table: 0.5,
	text: 0.5,
	title: 0.45,
	code: 0.45,
	checkbox_selected: 0.45,
	checkbox_unselected: 0.45,
	form: 0.45,
	key_value_region: 0.45,
	document_index: 0.45
};

const OVERLAP_PARAMS = {
	regular: { areaThreshold: 1.3, confThreshold: 0.05 },
	picture: { areaThreshold: 2.0, confThreshold: 0.3 },
	wrapper: { areaThreshold: 2.0, confThreshold: 0.2 }
} satisfies Record<string, OverlapParams>;

const TABLE_LABELS = new Set<LayoutLabel>(['table', 'document_index']);
const CONTAINER_LABELS = new Set<LayoutLabel>(['form', 'key_value_region']);
const BOX_KEEPING_LABELS = new Set<LayoutLabel>([
	'table',
	'checkbox_selected',
	'checkbox_unselected'
]);
const MIN_CELL_OVERLAP = 0.2;
const OVERLAP_THRESHOLD = 0.8;
const CHILD_CONTAINMENT = 0.8;
const MAX_PICTURE_PAGE_SHARE = 0.9;
const MAX_REFINEMENT_PASSES = 3;

function isSpecial(label: LayoutLabel): boolean {
	return label === 'picture' || TABLE_LABELS.has(label);
}

function sortItems(items: TextItemBox[]): TextItemBox[] {
	const unique = new Map(items.map((item) => [item.index, item]));
	return [...unique.values()].sort((a, b) => a.index - b.index);
}

function assignItems(clusters: LayoutCluster[], items: TextItemBox[]): void {
	for (const cluster of clusters) cluster.items = [];
	for (const item of items) {
		let best: LayoutCluster | undefined;
		let bestOverlap = MIN_CELL_OVERLAP;
		for (const cluster of clusters) {
			const overlap = intersectionOverSelf(item.box, cluster.box);
			if (overlap > bestOverlap) {
				bestOverlap = overlap;
				best = cluster;
			}
		}
		best?.items.push(item);
	}
}

function sameLine(line: TextItemBox[], item: TextItemBox): boolean {
	const last = line[line.length - 1];
	const lineBox = unionBox(line.map(({ box }) => box));
	const minHeight = Math.min(lineBox.b - lineBox.t, item.box.b - item.box.t);
	const gap = item.box.l - last.box.r;
	return (
		verticalOverlap(lineBox, item.box) > minHeight / 2 &&
		gap < Math.max(last.fontHeight, item.fontHeight) &&
		gap > -Math.max(last.fontHeight, item.fontHeight)
	);
}

function groupOrphanLines(items: TextItemBox[]): TextItemBox[][] {
	const lines: TextItemBox[][] = [];
	for (const item of items) {
		const line = lines.at(-1);
		if (line && sameLine(line, item)) line.push(item);
		else lines.push([item]);
	}
	return lines;
}

function adjustBoxes(clusters: LayoutCluster[]): void {
	for (const cluster of clusters) {
		if (cluster.items.length === 0) continue;
		const itemsBox = unionBox(cluster.items.map(({ box }) => box));
		cluster.box = BOX_KEEPING_LABELS.has(cluster.label)
			? unionBox([cluster.box, itemsBox])
			: itemsBox;
	}
}

function overlaps(a: LayoutCluster, b: LayoutCluster): boolean {
	return (
		intersectionOverUnion(a.box, b.box) > OVERLAP_THRESHOLD ||
		intersectionOverSelf(a.box, b.box) > OVERLAP_THRESHOLD ||
		intersectionOverSelf(b.box, a.box) > OVERLAP_THRESHOLD
	);
}

function shouldPrefer(
	candidate: LayoutCluster,
	other: LayoutCluster,
	params: OverlapParams
): boolean {
	const areaRatio = boxArea(candidate.box) / Math.max(boxArea(other.box), Number.EPSILON);
	if (candidate.label === 'list_item' && other.label === 'text' && Math.abs(1 - areaRatio) < 0.2) {
		return true;
	}
	if (candidate.label === 'code' && intersectionOverSelf(other.box, candidate.box) > 0.8) {
		return true;
	}
	const confDiff = other.confidence - candidate.confidence;
	return !(areaRatio <= params.areaThreshold && confDiff > params.confThreshold);
}

function selectBest(group: LayoutCluster[], params: OverlapParams): LayoutCluster {
	let best: LayoutCluster | undefined;
	for (const candidate of group) {
		const wins = group.every(
			(other) => other === candidate || shouldPrefer(candidate, other, params)
		);
		if (!wins) continue;
		if (
			!best ||
			(boxArea(candidate.box) > boxArea(best.box) &&
				best.confidence - candidate.confidence <= params.confThreshold)
		) {
			best = candidate;
		}
	}
	return best ?? group[0];
}

function removeOverlapping(clusters: LayoutCluster[], params: OverlapParams): LayoutCluster[] {
	const parent = clusters.map((_, index) => index);
	const find = (index: number): number => {
		while (parent[index] !== index) {
			parent[index] = parent[parent[index]];
			index = parent[index];
		}
		return index;
	};

	clusters.forEach((cluster, i) => {
		for (let j = i + 1; j < clusters.length; j += 1) {
			if (overlaps(cluster, clusters[j])) parent[find(j)] = find(i);
		}
	});

	const groups = new Map<number, LayoutCluster[]>();
	clusters.forEach((cluster, index) => {
		const root = find(index);
		groups.set(root, [...(groups.get(root) ?? []), cluster]);
	});

	return [...groups.values()].map((group) => {
		if (group.length === 1) return group[0];
		const best = selectBest(group, params);
		best.items = sortItems(group.flatMap((cluster) => cluster.items));
		return best;
	});
}

function processRegular(
	regular: LayoutCluster[],
	items: TextItemBox[],
	nextId: number
): LayoutCluster[] {
	let clusters = regular.filter(
		(cluster) => cluster.confidence >= CONFIDENCE_THRESHOLDS[cluster.label]
	);
	assignItems(clusters, items);
	clusters = clusters.filter((cluster) => cluster.items.length > 0 || cluster.label === 'formula');

	const assigned = new Set(clusters.flatMap((cluster) => cluster.items.map(({ index }) => index)));
	const orphans = items.filter((item) => !assigned.has(item.index));
	groupOrphanLines(orphans).forEach((line, offset) => {
		clusters.push({
			id: nextId + offset,
			label: 'text',
			confidence: 1,
			box: unionBox(line.map(({ box }) => box)),
			items: line,
			children: []
		});
	});

	let previousCount = clusters.length + 1;
	for (let pass = 0; pass < MAX_REFINEMENT_PASSES && previousCount !== clusters.length; pass += 1) {
		previousCount = clusters.length;
		adjustBoxes(clusters);
		clusters = removeOverlapping(clusters, OVERLAP_PARAMS.regular);
	}
	return clusters;
}

function dropCoincident(losers: LayoutCluster[], winners: LayoutCluster[]): Set<number> {
	const dropped = new Set<number>();
	for (const loser of losers) {
		const beaten = winners.some(
			(winner) =>
				intersectionOverUnion(loser.box, winner.box) > OVERLAP_THRESHOLD &&
				loser.confidence - winner.confidence < 0.1
		);
		if (beaten) dropped.add(loser.id);
	}
	return dropped;
}

function processSpecial(
	special: LayoutCluster[],
	regular: LayoutCluster[],
	page: PageSize
): LayoutCluster[] {
	const pageArea = page.width * page.height;
	const kept = special.filter(
		(cluster) =>
			cluster.confidence >= CONFIDENCE_THRESHOLDS[cluster.label] &&
			!(cluster.label === 'picture' && boxArea(cluster.box) / pageArea > MAX_PICTURE_PAGE_SHARE)
	);

	const pictures = removeOverlapping(
		kept.filter(({ label }) => label === 'picture'),
		OVERLAP_PARAMS.picture
	);
	const tables = removeOverlapping(
		kept.filter(({ label }) => TABLE_LABELS.has(label)),
		OVERLAP_PARAMS.wrapper
	);
	const plainTables = tables.filter(({ label }) => label === 'table');
	const indexes = tables.filter(({ label }) => label === 'document_index');
	const dropped = new Set([
		...dropCoincident(plainTables, indexes),
		...dropCoincident(pictures, tables)
	]);

	const wrappers = [...pictures, ...tables].filter(({ id }) => !dropped.has(id));
	for (const wrapper of wrappers) {
		wrapper.children = regular.filter(
			(child) => intersectionOverSelf(child.box, wrapper.box) > CHILD_CONTAINMENT
		);
		wrapper.items = sortItems(wrapper.children.flatMap((child) => child.items));
	}
	return wrappers;
}

function firstItemIndex(cluster: LayoutCluster): number {
	return cluster.items.length > 0 ? cluster.items[0].index : Number.MAX_SAFE_INTEGER;
}

export function postprocessLayout(
	detections: LayoutDetection[],
	items: TextItemBox[],
	page: PageSize
): LayoutCluster[] {
	const clusters: LayoutCluster[] = detections
		.filter(({ label }) => !CONTAINER_LABELS.has(label))
		.map((detection, id) => ({
			id,
			label: detection.label,
			confidence: detection.score,
			box: detection.box,
			items: [],
			children: []
		}));

	const regular = processRegular(
		clusters.filter(({ label }) => !isSpecial(label)),
		items,
		clusters.length
	);
	const wrappers = processSpecial(
		clusters.filter(({ label }) => isSpecial(label)),
		regular,
		page
	);
	const nested = new Set(wrappers.flatMap((wrapper) => wrapper.children.map(({ id }) => id)));

	return [...regular.filter(({ id }) => !nested.has(id)), ...wrappers]
		.map((cluster) => ({ ...cluster, items: sortItems(cluster.items) }))
		.sort(
			(a, b) => firstItemIndex(a) - firstItemIndex(b) || a.box.t - b.box.t || a.box.l - b.box.l
		);
}
