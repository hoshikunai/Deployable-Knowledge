import type { PdfDocument, PdfOutlineNode } from '../page-images';
import type { OutlineEntry } from './layout-types';

const MIN_ENTRIES = 3;
const PAGES_PER_ENTRY = 25;
const MAX_LABEL_LENGTH = 80;
const NUMBERED_TITLE = /^([A-Z]?\d+(?:\.\d+)*\.?)\s+(.+)$/;
const SENTENCE_END = /[.:](?=\s|$)/;
const FIGURE_OR_TABLE = /^(?:table|figure)\s+[A-Z]?\d/i;

type Target = Pick<OutlineEntry, 'pageIndex' | 'x' | 'y'>;

export function compactText(text: string): string {
	return text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
}

/**
 * The part of a section label its heading line must start with: the paragraph number for
 * numbered labels ("1.9."), otherwise the first two words ("Chapter 2"), compacted.
 */
export function labelHead(label: string): string {
	const numbered = NUMBERED_TITLE.exec(label);
	return compactText(numbered ? numbered[1] : label.split(' ').slice(0, 2).join(' '));
}

function truncateAtWord(text: string): string {
	if (text.length <= MAX_LABEL_LENGTH) return text;
	const cut = text.slice(0, MAX_LABEL_LENGTH);
	const boundary = cut.lastIndexOf(' ');
	return boundary > 0 ? cut.slice(0, boundary) : cut;
}

/**
 * Turns a bookmark title into a short section label. Generated bookmarks often hold the opening
 * of the paragraph itself ("1.1.  Scope.  This manual outlines…"), so numbered titles keep the
 * number and the first sentence only.
 */
export function sectionLabel(title: string): string {
	const text = title.replace(/\s+/g, ' ').trim();
	const numbered = NUMBERED_TITLE.exec(text);
	if (!numbered) return truncateAtWord(text);

	const [, number, rest] = numbered;
	const end = SENTENCE_END.exec(rest);
	const heading = end ? rest.slice(0, end.index) : rest;
	return truncateAtWord(`${number} ${heading}`.trim());
}

function coordinate(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

async function resolveTarget(
	document: PdfDocument,
	dest: PdfOutlineNode['dest']
): Promise<Target | null> {
	const explicit = typeof dest === 'string' ? await document.getDestination(dest) : dest;
	if (!Array.isArray(explicit) || explicit.length < 2) return null;

	const [ref, mode, ...args] = explicit;
	const pageIndex = Number.isInteger(ref)
		? (ref as number)
		: await document.getPageIndex(ref).catch(() => null);
	if (pageIndex === null || pageIndex < 0 || pageIndex >= document.numPages) return null;

	const name = typeof mode === 'object' && mode !== null && 'name' in mode ? mode.name : null;
	switch (name) {
		case 'XYZ':
			return { pageIndex, x: coordinate(args[0]), y: coordinate(args[1]) };
		case 'FitH':
		case 'FitBH':
			return { pageIndex, x: null, y: coordinate(args[0]) };
		case 'FitR':
			return { pageIndex, x: coordinate(args[0]), y: coordinate(args[3]) };
		default:
			return { pageIndex, x: null, y: null };
	}
}

/**
 * Reads the PDF's bookmark outline as section entries in outline order, with each target in PDF
 * user space. Returns nothing when the outline is too sparse to describe the document's
 * structure (cover-and-back-page bookmarks, for example). Table and figure bookmarks are left
 * out because they do not open sections.
 */
export async function readOutline(document: PdfDocument): Promise<OutlineEntry[]> {
	const nodes = await document.getOutline();
	if (!nodes) return [];

	const entries: OutlineEntry[] = [];
	const visit = async (items: PdfOutlineNode[], level: number) => {
		for (const node of items) {
			const label = sectionLabel(node.title);
			const target = await resolveTarget(document, node.dest);
			if (target && label && !FIGURE_OR_TABLE.test(label))
				entries.push({ level, label, ...target });
			await visit(node.items, level + 1);
		}
	};
	await visit(nodes, 1);

	const minimum = Math.max(MIN_ENTRIES, Math.ceil(document.numPages / PAGES_PER_ENTRY));
	return entries.length >= minimum ? entries : [];
}
