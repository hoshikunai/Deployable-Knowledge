import type { Box, CellLabel, TableCell, TableStructure } from './table-types';

const TAG_NAMES: Record<number, string> = {
	4: 'ecel',
	5: 'fcel',
	6: 'lcel',
	7: 'ucel',
	8: 'xcel',
	9: 'nl',
	10: 'ched',
	11: 'rhed',
	12: 'srow'
};

const CELL_LABELS = new Set<string>(['fcel', 'ecel', 'ched', 'rhed', 'srow']);
const RIGHT_STOP = new Set(['fcel', 'ched', 'rhed', 'srow', 'ecel', 'ucel']);
const DOWN_STOP = new Set(['fcel', 'ched', 'rhed', 'srow', 'ecel', 'lcel']);

function splitRows(tags: string[]): string[][] {
	const rows: string[][] = [];
	let current: string[] = [];
	for (const tag of tags) {
		if (tag !== 'nl') {
			current.push(tag);
			continue;
		}
		if (current.length > 0) rows.push(current);
		current = [];
	}
	if (current.length > 0) rows.push(current);

	const width = Math.max(...rows.map((row) => row.length));
	return rows.map((row) => [...row, ...Array<string>(width - row.length).fill('lcel')]);
}

function spanLength(cells: string[], start: number, stops: Set<string>): number {
	let length = 1;
	for (let index = start + 1; index < cells.length; index += 1) {
		if (stops.has(cells[index])) break;
		length += 1;
	}
	return length;
}

// Docling drops the box that follows each spanned cell when the box count and cell count
// disagree (TFPredictor._remove_bbox_span_desync)
function syncBoxes(boxes: Box[], spanned: boolean[]): Box[] {
	if (boxes.length === spanned.length) return boxes;
	const dropped = new Set<number>();
	spanned.forEach((isSpanned, index) => {
		if (isSpanned) dropped.add(index + 1);
	});
	return boxes.filter((_, index) => !dropped.has(index));
}

function toPageBox(values: Float32Array, index: number, table: Box): Box {
	const width = table.r - table.l;
	const height = table.b - table.t;
	const [l, t, r, b] = values.subarray(index * 4, index * 4 + 4);
	return {
		l: table.l + l * width,
		t: table.t + t * height,
		r: table.l + r * width,
		b: table.t + b * height
	};
}

/**
 * Turns TableFormer's OTSL tags into a grid, following Docling's otsl_to_html and
 * TFCellMatcher._build_table_cells: rows are padded to the widest row with lcel, a cell spans
 * right over the lcel run after it and down over the ucel run below it, and the n-th cell
 * takes the n-th predicted box.
 */
export function buildTableStructure(
	tagIds: Int32Array,
	normalizedBoxes: Float32Array,
	table: Box
): TableStructure | null {
	const rows = splitRows(Array.from(tagIds, (id) => TAG_NAMES[id] ?? 'unknown'));
	if (rows.length === 0 || rows[0].length === 0) return null;

	const cells: TableCell[] = [];
	const spanned: boolean[] = [];
	rows.forEach((row, rowIndex) => {
		row.forEach((tag, columnIndex) => {
			if (!CELL_LABELS.has(tag)) return;
			const column = rows.map((cellsInRow) => cellsInRow[columnIndex]);
			const columnSpan =
				row[columnIndex + 1] === 'lcel' ? spanLength(row, columnIndex, RIGHT_STOP) : 1;
			const rowSpan =
				rows[rowIndex + 1]?.[columnIndex] === 'ucel' ? spanLength(column, rowIndex, DOWN_STOP) : 1;
			spanned.push(columnSpan > 1 || rowSpan > 1);
			cells.push({
				row: rowIndex,
				column: columnIndex,
				rowSpan,
				columnSpan,
				label: tag as CellLabel,
				box: null,
				text: ''
			});
		});
	});

	const boxCount = normalizedBoxes.length / 4;
	const boxes = syncBoxes(
		Array.from({ length: boxCount }, (_, index) => toPageBox(normalizedBoxes, index, table)),
		spanned
	);
	cells.forEach((cell, index) => {
		cell.box = boxes[index] ?? null;
	});

	return { rowCount: rows.length, columnCount: rows[0].length, cells };
}
