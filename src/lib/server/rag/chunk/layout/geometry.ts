import type { Box } from './layout-types';

export function boxArea(box: Box): number {
	return Math.max(0, box.r - box.l) * Math.max(0, box.b - box.t);
}

export function intersectionArea(a: Box, b: Box): number {
	const width = Math.min(a.r, b.r) - Math.max(a.l, b.l);
	const height = Math.min(a.b, b.b) - Math.max(a.t, b.t);
	return width > 0 && height > 0 ? width * height : 0;
}

export function intersectionOverSelf(self: Box, other: Box): number {
	const area = boxArea(self);
	return area > 0 ? intersectionArea(self, other) / area : 0;
}

export function intersectionOverUnion(a: Box, b: Box): number {
	const intersection = intersectionArea(a, b);
	const union = boxArea(a) + boxArea(b) - intersection;
	return union > 0 ? intersection / union : 0;
}

export function unionBox(boxes: Box[]): Box {
	return {
		l: Math.min(...boxes.map((box) => box.l)),
		t: Math.min(...boxes.map((box) => box.t)),
		r: Math.max(...boxes.map((box) => box.r)),
		b: Math.max(...boxes.map((box) => box.b))
	};
}

export function verticalOverlap(a: Box, b: Box): number {
	return Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
}
