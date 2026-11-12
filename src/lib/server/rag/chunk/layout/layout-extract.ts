import { setImmediate as yieldEventLoop } from 'node:timers/promises';
import { createLogger } from '#lib/server/logging';
import { withTimeout } from '#lib/server/utils/with-timeout';
import type { PdfDocument, PdfViewport } from '../page-images';
import type { ExtractedChunk, SectionHeading, Source } from '../parse-shared';
import { buildLayoutChunks, type LayoutPageChunks } from './layout-chunks';
import type { OutlineEntry, SectionAnchor } from './layout-types';
import { LAYOUT_INPUT_SIZE, detectLayout } from './layout-worker';
import { readOutline } from './outline';
import { readTextItems, renderLayoutInput } from './page-input';
import { postprocessLayout } from './postprocess';
import { assignSectionPaths, type LayoutPage } from './section-paths';

const PAGE_RENDER_TIMEOUT_MS = 60_000;
const MIN_COVERAGE = 0.5;

const log = createLogger('documents');

function toAnchor(entry: OutlineEntry, viewport: PdfViewport): SectionAnchor {
	const anchor = { level: entry.level, label: entry.label };
	if (entry.y === null) return { ...anchor, x: 0, y: 0 };
	const [x, y] = viewport.convertToViewportPoint(entry.x ?? 0, entry.y);
	return { ...anchor, x: entry.x === null ? 0 : x, y };
}

async function analyzePage(
	document: PdfDocument,
	pageNumber: number,
	canvas: ReturnType<PdfDocument['canvasFactory']['create']>,
	source: Source,
	outline: OutlineEntry[] | null
): Promise<LayoutPageChunks | null> {
	const page = await document.getPage(pageNumber);
	try {
		const viewport = page.getViewport({ scale: 1 });
		const items = await readTextItems(page, viewport);
		if (items.length === 0) return null;

		const input = await withTimeout(
			renderLayoutInput(page, viewport, canvas),
			PAGE_RENDER_TIMEOUT_MS,
			`Page render timed out after ${PAGE_RENDER_TIMEOUT_MS / 1000}s.`
		);
		const detections = await detectLayout(input);
		const size = { width: viewport.width, height: viewport.height };
		const clusters = postprocessLayout(detections, items, size);
		const anchors = outline?.map((entry) => toAnchor(entry, viewport)) ?? null;
		const result = buildLayoutChunks(clusters, size, source, pageNumber - 1, anchors);

		if (result.coverage < MIN_COVERAGE) {
			log.warn('Layout analysis dropped too much page text; using plain extraction', {
				coverage: Math.round(result.coverage * 100),
				page: pageNumber
			});
			return null;
		}
		return result;
	} finally {
		page.cleanup();
	}
}

function asHeadings(entries: OutlineEntry[]): SectionHeading[] {
	return entries.map(({ label, level }) => ({ text: label, level }));
}

function outlineByPage(outline: OutlineEntry[]): Map<number, OutlineEntry[]> {
	const byPage = new Map<number, OutlineEntry[]>();
	for (const entry of outline) {
		byPage.set(entry.pageIndex, [...(byPage.get(entry.pageIndex) ?? []), entry]);
	}
	return byPage;
}

/**
 * Runs layout analysis page by page and returns the chunks for every page it handled, each
 * tagged with its section path. Sections come from the PDF's bookmarks when it has a usable
 * outline and from detected section headers otherwise. Pages that fail, time out or lose too
 * much text are left out so the caller extracts them the plain way; their bookmarks still
 * advance the open sections.
 */
export async function collectLayoutChunks(
	getDocument: () => PdfDocument,
	source: Source,
	pageNumbers: number[],
	onPage?: (current: number, total: number) => void
): Promise<Map<number, ExtractedChunk[]>> {
	const started = Date.now();
	const document = getDocument();
	const outline = await readOutline(document).catch((error: unknown) => {
		log.warn('Could not read the PDF outline; using detected headings', {}, error);
		return [];
	});
	const entriesByPage = outline.length > 0 ? outlineByPage(outline) : null;
	const pages: (LayoutPage & { pageNumber: number | null })[] = [];
	const canvas = document.canvasFactory.create(LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE);

	let previousPage = 0;

	try {
		for (const [position, pageNumber] of pageNumbers.entries()) {
			const entries = entriesByPage ? (entriesByPage.get(pageNumber - 1) ?? []) : null;
			const skipped: OutlineEntry[] = [];
			for (let index = previousPage; index < pageNumber - 1; index += 1) {
				skipped.push(...(entriesByPage?.get(index) ?? []));
			}
			if (skipped.length > 0) {
				pages.push({ pageNumber: null, chunks: [], trailingHeadings: asHeadings(skipped) });
			}
			previousPage = pageNumber;

			let result: LayoutPageChunks | null = null;
			try {
				result = await analyzePage(getDocument(), pageNumber, canvas, source, entries);
			} catch (error) {
				log.warn('Layout analysis failed; using plain extraction', { page: pageNumber }, error);
			}
			pages.push(
				result
					? { pageNumber, chunks: result.chunks, trailingHeadings: result.trailingHeadings }
					: {
							pageNumber: null,
							chunks: [],
							trailingHeadings: asHeadings(entries ?? [])
						}
			);
			onPage?.(position + 1, pageNumbers.length);
			await yieldEventLoop();
		}
	} finally {
		document.canvasFactory.destroy(canvas);
	}

	const chunksByPage = new Map<number, ExtractedChunk[]>();
	assignSectionPaths(pages).forEach((chunks, index) => {
		const { pageNumber } = pages[index];
		if (pageNumber !== null) chunksByPage.set(pageNumber, chunks);
	});

	log.debug('Layout analysis finished', {
		document: source.title,
		durationMs: Date.now() - started,
		pages: pageNumbers.length,
		structuredPages: chunksByPage.size,
		outlineEntries: outline.length
	});
	return chunksByPage;
}
