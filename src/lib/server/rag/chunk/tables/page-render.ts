import type { PdfDocument, PdfPage, PdfViewport } from '../page-images';
import type { Box, TextWord } from './table-types';
import { LAYOUT_INPUT_SIZE, type RgbaImage } from './table-worker';

// Docling renders pages at 2x and shrinks them to 1024 px tall before cropping tables
const TABLE_PAGE_HEIGHT_PX = 1024;

type PdfTextItem = {
	str: string;
	transform: number[];
	width: number;
	height: number;
};

function isTextItem(item: unknown): item is PdfTextItem {
	return typeof item === 'object' && item !== null && 'str' in item && 'transform' in item;
}

function applyTransform(matrix: number[], x: number, y: number): [number, number] {
	const [a, b, c, d, e, f] = matrix;
	return [a * x + c * y + e, b * x + d * y + f];
}

// Places a text run in viewport coordinates (top-left origin, scale 1) by pushing the corners
// of its glyph box through the run's text matrix and then the viewport transform
function itemBox(item: PdfTextItem, viewport: PdfViewport): Box {
	const [a, b, c, d, e, f] = item.transform;
	const advance = Math.hypot(a, b) || 1;
	const fontHeight = Math.hypot(c, d) || item.height || 1;
	const ux = (a / advance) * item.width;
	const uy = (b / advance) * item.width;
	const hasHeightAxis = Math.hypot(c, d) > 0;
	const vx = hasHeightAxis ? c : (-b / advance) * fontHeight;
	const vy = hasHeightAxis ? d : (a / advance) * fontHeight;
	const corners = [
		[e, f],
		[e + ux, f + uy],
		[e + vx, f + vy],
		[e + ux + vx, f + uy + vy]
	].map(([x, y]) => applyTransform(viewport.transform, x, y));
	const xs = corners.map(([x]) => x);
	const ys = corners.map(([, y]) => y);
	return { l: Math.min(...xs), t: Math.min(...ys), r: Math.max(...xs), b: Math.max(...ys) };
}

// A text run can cover several table columns, so horizontal runs are cut into words placed
// by their share of the run's characters
function splitIntoWords(text: string, box: Box, horizontal: boolean): TextWord[] {
	const spacedBefore = /^\s/.test(text);
	const spacedAfter = /\s$/.test(text);
	const matches = [...text.matchAll(/\S+/g)];
	if (!horizontal || matches.length <= 1) {
		return [{ text: text.trim(), box, spacedBefore, spacedAfter }];
	}

	const width = box.r - box.l;
	return matches.map((match, index) => {
		const start = match.index / text.length;
		const end = (match.index + match[0].length) / text.length;
		return {
			text: match[0],
			box: { l: box.l + start * width, t: box.t, r: box.l + end * width, b: box.b },
			spacedBefore: index > 0 || spacedBefore,
			spacedAfter: index < matches.length - 1 || spacedAfter
		};
	});
}

export async function readTextWords(page: PdfPage, viewport: PdfViewport): Promise<TextWord[]> {
	const content = await page.getTextContent();
	const words: TextWord[] = [];
	for (const item of content.items) {
		if (!isTextItem(item) || !item.str.trim()) continue;
		const box = itemBox(item, viewport);
		if (box.r <= box.l || box.b <= box.t) continue;
		const [a, b] = item.transform;
		words.push(...splitIntoWords(item.str, box, a > 0 && Math.abs(b) < 1e-6));
	}
	return words;
}

async function renderRegion(
	document: PdfDocument,
	page: PdfPage,
	viewport: PdfViewport,
	width: number,
	height: number,
	transform: number[]
): Promise<RgbaImage> {
	const canvas = document.canvasFactory.create(width, height);
	try {
		await page.render({
			canvas: canvas.canvas,
			canvasContext: canvas.context,
			viewport,
			transform
		}).promise;
		const { data } = canvas.context.getImageData(0, 0, width, height);
		return { pixels: new Uint8Array(data), width, height };
	} finally {
		document.canvasFactory.destroy(canvas);
	}
}

export function renderLayoutInput(
	document: PdfDocument,
	page: PdfPage,
	viewport: PdfViewport
): Promise<RgbaImage> {
	const size = LAYOUT_INPUT_SIZE;
	return renderRegion(document, page, viewport, size, size, [
		size / viewport.width,
		0,
		0,
		size / viewport.height,
		0,
		0
	]);
}

export function clampToPage(box: Box, viewport: PdfViewport): Box {
	return {
		l: Math.max(0, box.l),
		t: Math.max(0, box.t),
		r: Math.min(viewport.width, box.r),
		b: Math.min(viewport.height, box.b)
	};
}

/** Renders a table region at Docling's table scale and returns the crop and its exact box. */
export async function renderTableCrop(
	document: PdfDocument,
	page: PdfPage,
	viewport: PdfViewport,
	table: Box
): Promise<{ image: RgbaImage; box: Box }> {
	const scale = TABLE_PAGE_HEIGHT_PX / viewport.height;
	const left = Math.round(table.l * scale);
	const top = Math.round(table.t * scale);
	const width = Math.max(1, Math.round(table.r * scale) - left);
	const height = Math.max(1, Math.round(table.b * scale) - top);
	const image = await renderRegion(document, page, viewport, width, height, [
		scale,
		0,
		0,
		scale,
		-left,
		-top
	]);
	const box = {
		l: left / scale,
		t: top / scale,
		r: (left + width) / scale,
		b: (top + height) / scale
	};
	return { image, box };
}
