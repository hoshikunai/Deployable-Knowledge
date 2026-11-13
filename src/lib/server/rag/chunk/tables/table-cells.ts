import { normalizeWhitespace } from '../parse-shared';
import {
	intersectionOverSelf,
	type Box,
	type TableCell,
	type TableStructure,
	type TextWord
} from './table-types';

const MIN_WORD_IN_TABLE = 0.5;
const MIN_WORD_IN_CELL = 0.5;
const WORD_GAP_RATIO = 0.15;

function readingOrder(words: TextWord[]): string {
	const sorted = [...words].sort((a, b) => a.box.t - b.box.t);
	const lines: TextWord[][] = [];
	for (const word of sorted) {
		const line = lines.at(-1);
		const height = word.box.b - word.box.t;
		const lineCenter = line ? (line[0].box.t + line[0].box.b) / 2 : Number.NaN;
		const center = (word.box.t + word.box.b) / 2;
		if (line && Math.abs(center - lineCenter) < height / 2) line.push(word);
		else lines.push([word]);
	}
	return lines.map(joinLine).join(' ');
}

// pdf.js often splits one word into several runs ("I" + "dle"), so runs that touch (within a
// fraction of the text height either way) are joined without a space unless the PDF put a space
// between them; runs that overlap further are separate text stacked on the same line
function joinLine(line: TextWord[]): string {
	const sorted = [...line].sort((a, b) => a.box.l - b.box.l);
	let text = sorted[0].text;
	for (let index = 1; index < sorted.length; index += 1) {
		const previous = sorted[index - 1];
		const word = sorted[index];
		const gap = word.box.l - previous.box.r;
		const spaced =
			previous.spacedAfter ||
			word.spacedBefore ||
			Math.abs(gap) > (word.box.b - word.box.t) * WORD_GAP_RATIO;
		text += spaced ? ` ${word.text}` : word.text;
	}
	return text;
}

/**
 * Gives each cell the page words that fall mostly inside its predicted box, which is what
 * Docling does when cell matching is off (TableStructureOptions.do_cell_matching = False).
 */
export function fillCellText(structure: TableStructure, words: TextWord[], table: Box): void {
	const cellWords = new Map<TableCell, TextWord[]>();
	for (const word of words) {
		if (intersectionOverSelf(word.box, table) < MIN_WORD_IN_TABLE) continue;

		let best: TableCell | null = null;
		let bestOverlap = MIN_WORD_IN_CELL;
		for (const cell of structure.cells) {
			if (!cell.box) continue;
			const overlap = intersectionOverSelf(word.box, cell.box);
			if (overlap >= bestOverlap) {
				best = cell;
				bestOverlap = overlap;
			}
		}
		if (best) cellWords.set(best, [...(cellWords.get(best) ?? []), word]);
	}

	for (const [cell, assigned] of cellWords) {
		cell.text = normalizeWhitespace(readingOrder(assigned));
	}
}

function occupancyGrid(structure: TableStructure): (TableCell | undefined)[][] {
	const grid = Array.from({ length: structure.rowCount }, () =>
		Array<TableCell | undefined>(structure.columnCount)
	);
	for (const cell of structure.cells) {
		for (let row = cell.row; row < cell.row + cell.rowSpan; row += 1) {
			for (let column = cell.column; column < cell.column + cell.columnSpan; column += 1) {
				if (grid[row]) grid[row][column] = cell;
			}
		}
	}
	return grid;
}

function uniqueCells(row: (TableCell | undefined)[]): TableCell[] {
	return row.filter(
		(cell, index): cell is TableCell => cell !== undefined && row.indexOf(cell) === index
	);
}

function headerRowCount(grid: (TableCell | undefined)[][]): number {
	let count = 0;
	while (count < grid.length && grid[count].some((cell) => cell?.label === 'ched')) count += 1;
	return count;
}

function columnHeaders(grid: (TableCell | undefined)[][], headerRows: number): string[] {
	const width = grid[0]?.length ?? 0;
	return Array.from({ length: width }, (_, column) => {
		const parts: string[] = [];
		for (let row = 0; row < headerRows; row += 1) {
			const text = grid[row][column]?.text;
			if (text && parts.at(-1) !== text) parts.push(text);
		}
		return parts.join(' ');
	});
}

function rowLine(cells: TableCell[], headers: string[]): string {
	const filled = cells.filter((cell) => cell.text);
	if (filled.length === 0) return '';
	if (filled.length === 1 && filled[0].label === 'srow') return filled[0].text;
	if (headers.every((header) => !header)) return filled.map((cell) => cell.text).join(' | ');

	return filled
		.map((cell) => {
			const header = headers[cell.column];
			return header && header !== cell.text ? `${header}: ${cell.text}` : cell.text;
		})
		.join('; ');
}

/**
 * Writes a table as one line per body row. With column headers each value is labelled
 * ("Header: value"), so a chunk cut from the middle of a long table still says what its
 * numbers are; values that span rows are repeated on every row they cover.
 */
export function serializeTable(structure: TableStructure): string {
	const grid = occupancyGrid(structure);
	const headerRows = headerRowCount(grid);
	const headers = columnHeaders(grid, headerRows);
	const lines = grid
		.slice(headerRows)
		.map((row) => rowLine(uniqueCells(row), headers))
		.filter(Boolean);

	if (lines.length === 0) return headers.filter(Boolean).join(' | ');
	return lines.join('\n');
}
