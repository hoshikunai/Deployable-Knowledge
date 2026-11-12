import { RAG_CHUNK_CHARACTER_LIMIT } from '#lib/constants';
import { splitSentencesWithOffsets } from './chunker';
import { normalizeWhitespace, type ExtractedChunk } from './parse-shared';

const EDGE_LINES = 2;
const MIN_BANNER_PAGES = 3;
const BANNER_PAGE_RATIO = 0.3;
const MIN_PAGES_FOR_BANNERS = 4;
const MAX_CARRIED_FRAGMENT = RAG_CHUNK_CHARACTER_LIMIT / 2;
const CONTINUES_SENTENCE = /[a-z,\-­]\s*$/;
const HYPHENATED_END = /\p{L}[-­]\s*$/u;
const MAX_CARRIED_HEADING = 120;

type PageText = { first: number; last: number };

function textChunksByPage(chunks: ExtractedChunk[]): Map<number, PageText> {
	const pages = new Map<number, PageText>();
	chunks.forEach((chunk, index) => {
		if (chunk.chunkType !== 'TEXT') return;
		const page = pages.get(chunk.pageIndex);
		if (page) page.last = index;
		else pages.set(chunk.pageIndex, { first: index, last: index });
	});
	return pages;
}

function bannerKey(line: string): string {
	return line.toLowerCase().replace(/\s+/g, ' ').replace(/\d+/g, '#').trim();
}

function edgeLines(chunks: ExtractedChunk[], page: PageText): string[] {
	const head = chunks[page.first].content.split('\n').slice(0, EDGE_LINES);
	const tail = chunks[page.last].content.split('\n').slice(-EDGE_LINES);
	return [...head, ...tail];
}

function findBanners(chunks: ExtractedChunk[], pages: Map<number, PageText>): Set<string> {
	const pageCounts = new Map<string, number>();
	for (const page of pages.values()) {
		const keys = new Set(edgeLines(chunks, page).map(bannerKey).filter(Boolean));
		for (const key of keys) pageCounts.set(key, (pageCounts.get(key) ?? 0) + 1);
	}

	const minPages = Math.max(MIN_BANNER_PAGES, Math.ceil(pages.size * BANNER_PAGE_RATIO));
	return new Set([...pageCounts].filter(([, count]) => count >= minPages).map(([key]) => key));
}

function stripEdges(content: string, banners: Set<string>, edge: 'head' | 'tail'): string {
	const lines = content.split('\n');
	for (let step = 0; step < EDGE_LINES && lines.length > 0; step += 1) {
		const index = edge === 'head' ? 0 : lines.length - 1;
		if (!banners.has(bannerKey(lines[index]))) break;
		lines.splice(index, 1);
	}
	return lines.join('\n');
}

/**
 * Removes running headers and footers: lines at the top or bottom edge of a page that recur,
 * digits aside, on a large share of the document's pages.
 */
export function stripRepeatedPageLines(chunks: ExtractedChunk[]): ExtractedChunk[] {
	const pages = textChunksByPage(chunks);
	if (pages.size < MIN_PAGES_FOR_BANNERS) return chunks;

	const banners = findBanners(chunks, pages);
	if (banners.size === 0) return chunks;

	const cleaned = chunks.map((chunk) => ({ ...chunk }));
	for (const page of pages.values()) {
		cleaned[page.first].content = stripEdges(cleaned[page.first].content, banners, 'head');
		cleaned[page.last].content = stripEdges(cleaned[page.last].content, banners, 'tail');
	}
	return cleaned.filter((chunk) => chunk.chunkType !== 'TEXT' || chunk.content.trim());
}

/**
 * Moves a section heading stranded at the foot of a page onto the start of the next page's
 * text, so the heading stays with the body it introduces.
 */
export function carryTrailingHeadings(chunks: ExtractedChunk[]): ExtractedChunk[] {
	const pages = textChunksByPage(chunks);
	const carried = chunks.map((chunk) => ({ ...chunk }));
	const moved = new Set<number>();

	for (const [pageIndex, page] of pages) {
		const next = pages.get(pageIndex + 1);
		const heading = carried[page.last];
		const isHeadingOnly =
			heading.content.length <= MAX_CARRIED_HEADING && !heading.content.includes('\n');
		if (!next || !heading.opensSection || !isHeadingOnly) continue;

		const following = carried[next.first];
		following.content = `${heading.content}\n${following.content}`;
		following.opensSection = true;
		moved.add(page.last);
	}

	return carried.filter((_, index) => !moved.has(index));
}

function joinFragment(previous: string, fragment: string): string {
	const trimmed = previous.trimEnd();
	if (HYPHENATED_END.test(trimmed)) return trimmed.slice(0, -1) + fragment;
	return `${trimmed} ${fragment}`;
}

/**
 * Moves the opening fragment of a page onto the previous page when a sentence runs across the
 * page break, so it lands in one chunk attributed to the page where it starts.
 */
export function joinCrossPageSentences(chunks: ExtractedChunk[]): ExtractedChunk[] {
	const pages = textChunksByPage(chunks);
	const joined = chunks.map((chunk) => ({ ...chunk }));
	const emptied = new Set<number>();

	for (const [pageIndex, page] of pages) {
		const next = pages.get(pageIndex + 1);
		if (!next) continue;

		const previous = joined[page.last];
		const following = joined[next.first];
		if (following.opensSection || emptied.has(page.last)) continue;

		const previousText = previous.content.trimEnd();
		const followingText = normalizeWhitespace(following.content);
		if (!CONTINUES_SENTENCE.test(previousText)) continue;
		if (!/^\p{Ll}/u.test(followingText) && !HYPHENATED_END.test(previousText)) continue;

		const [fragment] = splitSentencesWithOffsets(followingText);
		if (!fragment || fragment.end - fragment.start > MAX_CARRIED_FRAGMENT) continue;

		previous.content = joinFragment(previousText, fragment.text);
		following.content = followingText.slice(fragment.end).trim();
		if (!following.content) emptied.add(next.first);
	}

	return joined.filter((_, index) => !emptied.has(index));
}
