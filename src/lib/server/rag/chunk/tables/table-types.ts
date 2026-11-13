export type Box = { l: number; t: number; r: number; b: number };

// spaced: the PDF itself put a space before or after this word
export type TextWord = { text: string; box: Box; spacedBefore: boolean; spacedAfter: boolean };

export type CellLabel = 'fcel' | 'ecel' | 'ched' | 'rhed' | 'srow';

export type TableCell = {
	row: number;
	column: number;
	rowSpan: number;
	columnSpan: number;
	label: CellLabel;
	box: Box | null;
	text: string;
};

export type TableStructure = {
	rowCount: number;
	columnCount: number;
	cells: TableCell[];
};

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
