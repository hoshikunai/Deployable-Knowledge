import type { PdfCanvasAndContext, PdfPage, PdfViewport } from '../page-images';
import type { Box, LayoutInput, TextItemBox } from './layout-types';
import { LAYOUT_INPUT_SIZE } from './layout-worker';

type PdfTextItem = {
	str: string;
	transform: number[];
	width: number;
	height: number;
	hasEOL: boolean;
};

function isTextItem(item: unknown): item is PdfTextItem {
	return typeof item === 'object' && item !== null && 'str' in item && 'transform' in item;
}

function applyTransform(matrix: number[], x: number, y: number): [number, number] {
	const [a, b, c, d, e, f] = matrix;
	return [a * x + c * y + e, b * x + d * y + f];
}

// Places a text run in viewport coordinates (top-left origin, scale 1) by pushing the four
// corners of its glyph box through the run's own text matrix and then the viewport transform,
// which keeps rotated text, rotated pages and offset crop boxes in the same space as the model.
function itemBox(item: PdfTextItem, viewport: PdfViewport): { box: Box; fontHeight: number } {
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
	return {
		box: { l: Math.min(...xs), t: Math.min(...ys), r: Math.max(...xs), b: Math.max(...ys) },
		fontHeight
	};
}

export async function readTextItems(page: PdfPage, viewport: PdfViewport): Promise<TextItemBox[]> {
	const content = await page.getTextContent();
	const items: TextItemBox[] = [];

	for (const raw of content.items) {
		if (!isTextItem(raw)) continue;
		const previous = items.at(-1);
		if (!raw.str.trim()) {
			if (raw.hasEOL && previous) previous.hasEOL = true;
			continue;
		}

		const { box, fontHeight } = itemBox(raw, viewport);
		if (box.r <= box.l || box.b <= box.t) continue;
		items.push({ index: items.length, text: raw.str, box, hasEOL: raw.hasEOL, fontHeight });
	}

	return items;
}

export async function renderLayoutInput(
	page: PdfPage,
	viewport: PdfViewport,
	canvas: PdfCanvasAndContext
): Promise<LayoutInput> {
	const scale = [
		LAYOUT_INPUT_SIZE / viewport.width,
		0,
		0,
		LAYOUT_INPUT_SIZE / viewport.height,
		0,
		0
	];
	await page.render({
		canvas: canvas.canvas,
		canvasContext: canvas.context,
		viewport,
		transform: scale
	}).promise;

	const { data } = canvas.context.getImageData(0, 0, LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE);
	return { pixels: new Uint8Array(data), width: viewport.width, height: viewport.height };
}
