import { setImmediate as yieldEventLoop } from 'node:timers/promises';
import { createLogger } from '#lib/server/logging';
import { withTimeout } from '#lib/server/utils/with-timeout';
import type { PdfDocument } from '../page-images';
import type { ExtractedChunk, Source } from '../parse-shared';
import { buildTableStructure } from './otsl';
import { clampToPage, readTextWords, renderLayoutInput, renderTableCrop } from './page-render';
import { fillCellText, serializeTable } from './table-cells';
import { intersectionOverSelf, type Box } from './table-types';
import { detectTables, recognizeTable, TableModelLoadError } from './table-worker';

const RENDER_TIMEOUT_MS = 60_000;
const NESTED_TABLE_OVERLAP = 0.8;

const log = createLogger('documents');

function distinctTables(tables: { box: Box; score: number }[]): Box[] {
	const kept: Box[] = [];
	for (const { box } of [...tables].sort((a, b) => b.score - a.score)) {
		if (kept.every((other) => intersectionOverSelf(box, other) < NESTED_TABLE_OVERLAP)) {
			kept.push(box);
		}
	}
	return kept.sort((a, b) => a.t - b.t || a.l - b.l);
}

function rendered<T>(task: Promise<T>): Promise<T> {
	return withTimeout(
		task,
		RENDER_TIMEOUT_MS,
		`Page render timed out after ${RENDER_TIMEOUT_MS / 1000}s.`
	);
}

async function readPageTables(
	document: PdfDocument,
	pageNumber: number,
	source: Source
): Promise<ExtractedChunk[]> {
	const page = await document.getPage(pageNumber);
	try {
		const viewport = page.getViewport({ scale: 1 });
		const layout = await rendered(renderLayoutInput(document, page, viewport));
		const tables = distinctTables(await detectTables(layout, viewport.width, viewport.height));
		if (tables.length === 0) return [];

		const words = await readTextWords(page, viewport);
		const chunks: ExtractedChunk[] = [];
		for (const table of tables) {
			const crop = await rendered(
				renderTableCrop(document, page, viewport, clampToPage(table, viewport))
			);
			let content = '';
			try {
				const { tags, boxes } = await recognizeTable(crop.image);
				const structure = buildTableStructure(tags, boxes, crop.box);
				if (structure) {
					fillCellText(structure, words, crop.box);
					content = serializeTable(structure);
				}
			} catch (error) {
				if (error instanceof TableModelLoadError) throw error;
				log.warn(
					'Could not read a table; its text stays in the page text',
					{
						page: pageNumber
					},
					error
				);
			}
			if (content) {
				chunks.push({
					chunkType: 'TABLE',
					source,
					pageIndex: pageNumber - 1,
					content,
					rowPerLine: true
				});
			}
		}
		return chunks;
	} finally {
		page.cleanup();
	}
}

/**
 * Finds tables with Heron and reads them with TableFormer, page by page. Returns the TABLE
 * chunks of every page it handled, including pages without tables; pages that fail are left
 * out so the caller can fall back to pdf-parse's table detection for them.
 */
export async function collectRecognizedTables(
	getDocument: () => PdfDocument,
	source: Source,
	pageNumbers: number[],
	onPage?: (current: number, total: number) => void
): Promise<Map<number, ExtractedChunk[]>> {
	const started = Date.now();
	const tablesByPage = new Map<number, ExtractedChunk[]>();

	for (const [position, pageNumber] of pageNumbers.entries()) {
		try {
			tablesByPage.set(pageNumber, await readPageTables(getDocument(), pageNumber, source));
		} catch (error) {
			if (error instanceof TableModelLoadError) {
				log.error('Table recognition is unavailable; using pdf-parse tables', {}, error);
				break;
			}
			log.warn(
				'Table recognition failed on a page; using pdf-parse tables',
				{
					page: pageNumber
				},
				error
			);
		}
		onPage?.(position + 1, pageNumbers.length);
		await yieldEventLoop();
	}

	log.debug('Table recognition finished', {
		document: source.title,
		durationMs: Date.now() - started,
		pages: pageNumbers.length,
		handledPages: tablesByPage.size,
		tables: [...tablesByPage.values()].reduce((sum, chunks) => sum + chunks.length, 0)
	});
	return tablesByPage;
}
