// Pure text transforms behind the notebook editor ribbon. Each command takes the
// current text plus selection and returns the next text with the selection to
// restore, so the component only moves the caret.

import type { NotebookHighlight, NotebookTextColor } from '$lib/constants';
import {
	formatTextStyle,
	skipBlockPrefix,
	styledSpanAt,
	wrapStyledLines,
	type NotebookTextStyle,
	type StyledSpanMatch
} from '$lib/utils/notebook-styles';

export interface EditorSelection {
	start: number;
	end: number;
}

export interface FormatResult {
	text: string;
	start: number;
	end: number;
}

export type InlineFormat = 'bold' | 'italic' | 'strikethrough' | 'code';
export type LineFormat = 'h1' | 'h2' | 'h3' | 'bullet-list' | 'ordered-list' | 'quote';
export type RibbonCommand =
	| InlineFormat
	| LineFormat
	| 'link'
	| 'code-block'
	| { color: NotebookTextColor }
	| { highlight: NotebookHighlight };

const INLINE_MARKERS: Record<InlineFormat, string> = {
	bold: '**',
	italic: '*',
	strikethrough: '~~',
	code: '`'
};

const HEADING_LEVELS: Record<'h1' | 'h2' | 'h3', number> = { h1: 1, h2: 2, h3: 3 };

// Markers have to sit flush against the text they wrap — CommonMark renders
// `** bold **` as literal asterisks — so whitespace the selection picked up (a
// double-click usually grabs the trailing space) stays outside the markup.
function trimSelection(text: string, { start, end }: EditorSelection): EditorSelection {
	const selected = text.slice(start, end);
	if (!selected.trim()) return { start: end, end };
	return {
		start: start + (selected.length - selected.trimStart().length),
		end: end - (selected.length - selected.trimEnd().length)
	};
}

export function toggleInline(
	text: string,
	selection: EditorSelection,
	format: InlineFormat
): FormatResult {
	const marker = INLINE_MARKERS[format];
	const { start, end } = trimSelection(text, selection);
	const selected = text.slice(start, end);
	const before = text.slice(0, start);
	const after = text.slice(end);

	if (before.endsWith(marker) && after.startsWith(marker)) {
		return {
			text: before.slice(0, -marker.length) + selected + after.slice(marker.length),
			start: start - marker.length,
			end: end - marker.length
		};
	}
	if (
		selected.length >= marker.length * 2 &&
		selected.startsWith(marker) &&
		selected.endsWith(marker)
	) {
		const inner = selected.slice(marker.length, selected.length - marker.length);
		return { text: before + inner + after, start, end: start + inner.length };
	}
	return {
		text: before + marker + selected + marker + after,
		start: start + marker.length,
		end: end + marker.length
	};
}

function lineBlock(text: string, { start, end }: EditorSelection) {
	const blockStart = text.lastIndexOf('\n', Math.max(0, start - 1)) + 1;
	const terminator = text.indexOf('\n', Math.max(end, start));
	const blockEnd = terminator === -1 ? text.length : terminator;
	return { blockStart, blockEnd, lines: text.slice(blockStart, blockEnd).split('\n') };
}

function replaceBlock(
	text: string,
	blockStart: number,
	blockEnd: number,
	lines: string[]
): FormatResult {
	const block = lines.join('\n');
	return {
		text: text.slice(0, blockStart) + block + text.slice(blockEnd),
		start: blockStart,
		end: blockStart + block.length
	};
}

export function toggleLinePrefix(
	text: string,
	selection: EditorSelection,
	format: LineFormat
): FormatResult {
	const { blockStart, blockEnd, lines } = lineBlock(text, selection);
	const content = lines.filter((line) => line.trim());

	if (format === 'h1' || format === 'h2' || format === 'h3') {
		const prefix = `${'#'.repeat(HEADING_LEVELS[format])} `;
		const active = content.length > 0 && content.every((line) => line.startsWith(prefix));
		const next = lines.map((line) => {
			const stripped = line.replace(/^#{1,6}\s+/, '');
			return active || !stripped.trim() ? stripped : prefix + stripped;
		});
		return replaceBlock(text, blockStart, blockEnd, next);
	}

	if (format === 'bullet-list') {
		const active = content.length > 0 && content.every((line) => /^[-*+]\s/.test(line));
		const next = lines.map((line) => {
			const stripped = line.replace(/^[-*+]\s+/, '');
			return active || !stripped.trim() ? stripped : `- ${stripped}`;
		});
		return replaceBlock(text, blockStart, blockEnd, next);
	}

	if (format === 'ordered-list') {
		const active = content.length > 0 && content.every((line) => /^\d+\.\s/.test(line));
		let ordinal = 0;
		const next = lines.map((line) => {
			const stripped = line.replace(/^\d+\.\s+/, '');
			if (active || !stripped.trim()) return stripped;
			ordinal += 1;
			return `${ordinal}. ${stripped}`;
		});
		return replaceBlock(text, blockStart, blockEnd, next);
	}

	const active = content.length > 0 && content.every((line) => /^>\s?/.test(line));
	const next = lines.map((line) => {
		const stripped = line.replace(/^>\s?/, '');
		return active || !stripped.trim() ? stripped : `> ${stripped}`;
	});
	return replaceBlock(text, blockStart, blockEnd, next);
}

export function insertLink(text: string, { start, end }: EditorSelection): FormatResult {
	const selected = text.slice(start, end) || 'link text';
	const url = 'https://';
	const snippet = `[${selected}](${url})`;
	const urlStart = start + selected.length + 3;
	return {
		text: text.slice(0, start) + snippet + text.slice(end),
		start: urlStart,
		end: urlStart + url.length
	};
}

function restyleSpan(text: string, span: StyledSpanMatch, style: NotebookTextStyle): FormatResult {
	const next: NotebookTextStyle = { ...span.style };
	if (style.color) next.color = span.style.color === style.color ? undefined : style.color;
	if (style.highlight) {
		next.highlight = span.style.highlight === style.highlight ? undefined : style.highlight;
	}
	const content = text.slice(span.open + 1, span.close);
	if (!next.color && !next.highlight) {
		return {
			text: text.slice(0, span.open) + content + text.slice(span.end),
			start: span.open,
			end: span.open + content.length
		};
	}
	return {
		text: `${text.slice(0, span.close + 1)}{${formatTextStyle(next)}}${text.slice(span.end)}`,
		start: span.open + 1,
		end: span.close
	};
}

// Restyles the span under the caret or selection (choosing its current color
// again removes it), otherwise wraps the selection in a new styled span
export function applyTextStyle(
	text: string,
	selection: EditorSelection,
	style: NotebookTextStyle
): FormatResult {
	const trimmed = trimSelection(text, selection);
	const span = styledSpanAt(text, trimmed.start, trimmed.end);
	if (span) return restyleSpan(text, span, style);

	const start = Math.min(skipBlockPrefix(text, trimmed.start), trimmed.end);
	const selected =
		text.slice(start, trimmed.end) || (style.color ? 'colored text' : 'highlighted text');
	const wrapped = wrapStyledLines(selected, style, { override: true });
	const plain = wrapped === `[${selected}]{${formatTextStyle(style)}}`;
	return {
		text: text.slice(0, start) + wrapped + text.slice(trimmed.end),
		start: plain ? start + 1 : start,
		end: plain ? start + 1 + selected.length : start + wrapped.length
	};
}

export function toggleCodeBlock(text: string, selection: EditorSelection): FormatResult {
	const { blockStart, blockEnd, lines } = lineBlock(text, selection);
	const fenced =
		lines.length >= 2 && lines[0].trim().startsWith('```') && lines.at(-1)?.trim() === '```';
	const next = fenced ? lines.slice(1, -1) : ['```', ...lines, '```'];
	return replaceBlock(text, blockStart, blockEnd, next);
}

export function applyRibbonCommand(
	text: string,
	selection: EditorSelection,
	command: RibbonCommand
): FormatResult {
	if (typeof command === 'object') return applyTextStyle(text, selection, command);
	if (command === 'link') return insertLink(text, selection);
	if (command === 'code-block') return toggleCodeBlock(text, selection);
	if (command in INLINE_MARKERS) return toggleInline(text, selection, command as InlineFormat);
	return toggleLinePrefix(text, selection, command as LineFormat);
}
