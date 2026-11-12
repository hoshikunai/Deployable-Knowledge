import { SECTION_PATH_SEPARATOR } from '../../chunk-passage';
import type { ExtractedChunk, SectionHeading } from '../parse-shared';

const MAX_DEPTH = 4;
const MAX_PATH_LENGTH = 200;
const MAX_HEADING_LENGTH = 120;
const MIN_HEADING_LETTERS = 2;
const KEYWORD_HEADING = /^(?:chapter|part|section|attachment|appendix|annex)\s+[\dA-Z]/i;
const NUMBERED_HEADING = /^(\d+(?:\.\d+)*)\.?\s/;

export type LayoutPage = {
	chunks: ExtractedChunk[];
	trailingHeadings: SectionHeading[];
};

type OpenSection = { level: number; text: string };

function isHeadingText(text: string): boolean {
	const letters = text.match(/\p{L}/gu)?.length ?? 0;
	return text.length <= MAX_HEADING_LENGTH && letters >= MIN_HEADING_LETTERS;
}

function fontRanks(pages: LayoutPage[]): Map<number, number> {
	const heights = new Set<number>();
	for (const page of pages) {
		const headings = [
			...page.chunks.flatMap((chunk) => chunk.headings ?? []),
			...page.trailingHeadings
		];
		for (const { fontHeight } of headings) {
			if (fontHeight !== undefined) heights.add(Math.round(fontHeight));
		}
	}
	return new Map([...heights].sort((a, b) => b - a).map((height, rank) => [height, rank]));
}

// Bookmarks carry their own depth. Detected headings have none, so chapter-style keywords open
// the top level, dotted numbers nest by their number of parts, and anything else ranks by font
// size among the document's headings, largest first.
function headingLevel(heading: SectionHeading, ranks: Map<number, number>): number {
	if (heading.level !== undefined) return heading.level;
	if (KEYWORD_HEADING.test(heading.text)) return 1;
	const numbered = NUMBERED_HEADING.exec(heading.text);
	if (numbered) return numbered[1].split('.').length;
	return (ranks.get(Math.round(heading.fontHeight ?? 0)) ?? ranks.size) + 1;
}

function formatPath(sections: OpenSection[]): string | undefined {
	if (sections.length === 0) return undefined;
	const texts = sections.map(({ text }) => text);
	const parts = texts.length > MAX_DEPTH ? [texts[0], ...texts.slice(1 - MAX_DEPTH)] : texts;
	while (parts.length > 2 && parts.join(SECTION_PATH_SEPARATOR).length > MAX_PATH_LENGTH) {
		parts.splice(1, 1);
	}
	return parts.join(SECTION_PATH_SEPARATOR).slice(0, MAX_PATH_LENGTH);
}

/**
 * Walks the structured pages in order with a stack of open sections and gives every chunk the
 * path of headings it sits under ("Chapter 1 > 1.4. Preparing… > 1.4.2. Suspenses"). A heading
 * closes every open section at its level or deeper before it opens. Trailing headings apply
 * after their page, so they reach the next page's chunks.
 */
export function assignSectionPaths(pages: LayoutPage[]): ExtractedChunk[][] {
	const ranks = fontRanks(pages);
	const open: OpenSection[] = [];

	const apply = (headings: SectionHeading[]) => {
		for (const heading of headings) {
			if (!isHeadingText(heading.text)) continue;
			const level = headingLevel(heading, ranks);
			while (open.length > 0 && open[open.length - 1].level >= level) open.pop();
			open.push({ level, text: heading.text });
		}
	};

	return pages.map((page) => {
		const chunks = page.chunks.map(({ headings, ...chunk }) => {
			apply(headings ?? []);
			const sectionPath = formatPath(open);
			return sectionPath ? { ...chunk, sectionPath } : chunk;
		});
		apply(page.trailingHeadings);
		return chunks;
	});
}
