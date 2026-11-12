import {
	normalizeWhitespace,
	type ExtractedChunk,
	type SectionHeading,
	type Source
} from '../parse-shared';
import { verticalOverlap } from './geometry';
import type {
	LayoutCluster,
	LayoutLabel,
	PageSize,
	SectionAnchor,
	TextItemBox
} from './layout-types';
import { compactText, labelHead, sectionLabel } from './outline';
import { orderClusters } from './reading-order';

const SECTION_LABELS = new Set<LayoutLabel>(['section_header', 'title']);
const TABLE_LABELS = new Set<LayoutLabel>(['table', 'document_index']);
const FURNITURE_LABELS = new Set<LayoutLabel>(['page_header', 'page_footer']);
const WORD_GAP_RATIO = 0.15;
const CELL_GAP_RATIO = 0.5;
const DOT_LEADER = /(?:\s?\.){4,}/g;
const ANCHOR_RISE = 4;
const ANCHOR_INDENT = 20;
const SAME_LINE_TOLERANCE = 1;
const LINE_HEAD_ITEMS = 3;
const BARE_KEYWORD_LABEL = /^(?:chapter|part|section|attachment|appendix|annex)\s+\S+$/i;

export type LayoutPageChunks = {
	chunks: ExtractedChunk[];
	coverage: number;
	trailingHeadings: SectionHeading[];
};

type AnchorPlacement = {
	splits: Map<number, SectionHeading[]>;
	unplaced: SectionHeading[];
};

type Segment = { opensSection: boolean; headings: SectionHeading[]; hasBody: boolean };

function needsSpace(previous: TextItemBox, item: TextItemBox): boolean {
	if (/\s$/.test(previous.text) || /^\s/.test(item.text)) return false;
	if (previous.hasEOL) return true;
	return item.box.l - previous.box.r > WORD_GAP_RATIO * Math.max(previous.fontHeight, 1);
}

function joinItems(items: TextItemBox[]): string {
	let text = '';
	items.forEach((item, index) => {
		const previous = items[index - 1];
		if (previous?.hasEOL && /\p{L}[-­]$/u.test(text) && /^\p{Ll}/u.test(item.text)) {
			text = text.slice(0, -1) + item.text;
			return;
		}
		text += previous && needsSpace(previous, item) ? ` ${item.text}` : item.text;
	});
	return normalizeWhitespace(text.replace(/\n/g, ' ').replace(DOT_LEADER, ' '));
}

function groupRows(items: TextItemBox[]): TextItemBox[][] {
	const rows: TextItemBox[][] = [];
	const byTop = [...items].sort((a, b) => a.box.t - b.box.t || a.box.l - b.box.l);
	for (const item of byTop) {
		const row = rows.find((candidate) =>
			candidate.some((member) => {
				const minHeight = Math.min(member.box.b - member.box.t, item.box.b - item.box.t);
				return verticalOverlap(member.box, item.box) > minHeight / 2;
			})
		);
		if (row) row.push(item);
		else rows.push([item]);
	}
	return rows.map((row) => row.sort((a, b) => a.box.l - b.box.l));
}

function rowCells(row: TextItemBox[]): string[] {
	const cells: TextItemBox[][] = [];
	for (const item of row) {
		const cell = cells.at(-1);
		const last = cell?.at(-1);
		if (cell && last && item.box.l - last.box.r < CELL_GAP_RATIO * Math.max(last.fontHeight, 1)) {
			cell.push(item);
		} else {
			cells.push([item]);
		}
	}
	return cells.map(joinItems).filter(Boolean);
}

function tableText(cluster: LayoutCluster): string {
	return groupRows(cluster.items)
		.map((row) => rowCells(row).join(' | '))
		.filter(Boolean)
		.join('\n');
}

function characterCount(clusters: LayoutCluster[]): number {
	return clusters.reduce(
		(sum, cluster) =>
			sum + cluster.items.reduce((total, item) => total + item.text.trim().length, 0),
		0
	);
}

function isCloserAnchorItem(item: TextItemBox, best: TextItemBox | undefined): boolean {
	if (!best) return true;
	const rise = best.box.t - item.box.t;
	if (rise > SAME_LINE_TOLERANCE) return true;
	return rise >= -SAME_LINE_TOLERANCE && item.index < best.index;
}

function lineHead(items: TextItemBox[], position: number): string {
	return compactText(
		items
			.slice(position, position + LINE_HEAD_ITEMS)
			.map(({ text }) => text)
			.join(' ')
	);
}

// A bookmark points at the top-left of its heading, so the heading is the first line at or just
// below that point and not left of it. Some bookmarks only point at the top of the page their
// section starts on, so a line that begins with the label's number or first words wins over a
// line that is merely closer.
function placeAnchors(anchors: SectionAnchor[], items: TextItemBox[]): AnchorPlacement {
	const splits = new Map<number, SectionHeading[]>();
	const unplaced: SectionHeading[] = [];

	for (const anchor of anchors) {
		const heading = { text: anchor.label, level: anchor.level };
		const head = labelHead(anchor.label);
		let nearest: TextItemBox | undefined;
		let matching: TextItemBox | undefined;
		items.forEach((item, position) => {
			if (item.box.t < anchor.y - ANCHOR_RISE || item.box.l < anchor.x - ANCHOR_INDENT) return;
			if (isCloserAnchorItem(item, nearest)) nearest = item;
			if (
				head &&
				lineHead(items, position).startsWith(head) &&
				isCloserAnchorItem(item, matching)
			) {
				matching = item;
			}
		});
		const placed = matching ?? nearest;
		if (placed) splits.set(placed.index, [...(splits.get(placed.index) ?? []), heading]);
		else unplaced.push(heading);
	}

	return { splits, unplaced };
}

function nameHeadings(headings: SectionHeading[], text: string): SectionHeading[] {
	const compact = compactText(text);
	return headings.map((heading) =>
		compact.startsWith(compactText(heading.text))
			? { ...heading, text: sectionLabel(text) }
			: heading
	);
}

// Chapter bookmarks are often just "Chapter 2", with the chapter's name set as its own heading
// line underneath; that line names the chapter instead of opening a section of its own.
function extendBareLabel(segment: Segment, text: string): boolean {
	const last = segment.headings.at(-1);
	if (segment.hasBody || !last || !BARE_KEYWORD_LABEL.test(last.text)) return false;
	segment.headings = [
		...segment.headings.slice(0, -1),
		{ ...last, text: sectionLabel(`${last.text} ${text}`) }
	];
	return true;
}

function clusterHeadings(cluster: LayoutCluster, splits: Map<number, SectionHeading[]>) {
	return cluster.items.flatMap((item) => splits.get(item.index) ?? []);
}

/**
 * Turns one page's clusters into extracted chunks in reading order. Section headers and
 * bookmark anchors open a new text segment so chunks never straddle sections, tables become
 * their own chunks, and running headers, footers and text inside pictures are left out.
 *
 * With `anchors` (the page's bookmark targets), segments carry the bookmark headings and a text
 * block is split where an anchor lands inside it. Without them (`null`), each detected section
 * header becomes the segment's heading. Anchors that land below all text are returned as
 * trailing headings for the caller to apply after the page. Coverage is the share of the page's
 * body text that made it into a chunk, used by the caller to fall back to plain extraction.
 */
export function buildLayoutChunks(
	clusters: LayoutCluster[],
	page: PageSize,
	source: Source,
	pageIndex: number,
	anchors: SectionAnchor[] | null
): LayoutPageChunks {
	const body = clusters.filter(({ label }) => !FURNITURE_LABELS.has(label));
	const ordered = orderClusters(body, page).filter(({ label }) => label !== 'picture');
	const { splits, unplaced } = placeAnchors(
		anchors ?? [],
		ordered.flatMap(({ items }) => items)
	);
	const chunks: ExtractedChunk[] = [];
	let lines: string[] = [];
	let segment: Segment = { opensSection: false, headings: [], hasBody: false };
	let emitted = 0;

	const withHeadings = (headings: SectionHeading[]) => (headings.length > 0 ? { headings } : {});

	const flush = () => {
		if (lines.length === 0) return;
		chunks.push({
			chunkType: 'TEXT',
			source,
			pageIndex,
			content: lines.join('\n'),
			opensSection: segment.opensSection,
			...withHeadings(segment.headings)
		});
		lines = [];
		segment = { opensSection: false, headings: [], hasBody: false };
	};

	const openSection = (headings: SectionHeading[]) => {
		flush();
		segment = { opensSection: true, headings: [...segment.headings, ...headings], hasBody: false };
	};

	for (const cluster of ordered) {
		if (TABLE_LABELS.has(cluster.label)) {
			const content = tableText(cluster);
			if (!content) continue;
			flush();
			const headings = [...segment.headings, ...clusterHeadings(cluster, splits)];
			chunks.push({ chunkType: 'TABLE', source, pageIndex, content, ...withHeadings(headings) });
			segment = { opensSection: false, headings: [], hasBody: false };
			emitted += characterCount([cluster]);
			continue;
		}

		if (SECTION_LABELS.has(cluster.label)) {
			const text = joinItems(cluster.items);
			if (!text) continue;
			const fontHeight = Math.max(...cluster.items.map((item) => item.fontHeight));
			const own =
				anchors === null
					? [{ text, fontHeight }]
					: nameHeadings(clusterHeadings(cluster, splits), text);
			if (own.length > 0 || !extendBareLabel(segment, text)) openSection(own);
			lines.push(text);
			emitted += characterCount([cluster]);
			continue;
		}

		let run: TextItemBox[] = [];
		for (const item of cluster.items) {
			const headings = splits.get(item.index);
			if (headings) {
				const text = joinItems(run);
				if (text) lines.push(text);
				run = [];
				openSection(headings);
			}
			run.push(item);
		}
		const text = joinItems(run);
		if (text) lines.push(text);
		segment.hasBody = true;
		emitted += characterCount([cluster]);
	}
	flush();

	const bodyCharacters = characterCount(ordered);
	return {
		chunks,
		coverage: bodyCharacters > 0 ? emitted / bodyCharacters : 1,
		trailingHeadings: [...segment.headings, ...unplaced]
	};
}
