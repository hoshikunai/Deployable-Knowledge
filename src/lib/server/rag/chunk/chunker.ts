import {
	buildChunkId,
	countWords,
	normalizeWhitespace,
	type ExtractedChunk,
	type ParsedChunk
} from './parse-shared';
import { chunkingConfig } from './chunking-config';
import { getEmbeddingTokenCounter } from '../embedding-model';

const MIN_WORDS = 5;

// Offsets point into the cleaned page text, not the original PDF bytes
type SentenceSpan = {
	text: string;
	start: number;
	end: number;
};

// How chunks are sized: by characters (the default) or by embedding-model tokens
export type ChunkBudget = {
	// Splits a range that cannot fit in one chunk into bounded spans
	splitRange(text: string, start: number, end: number): SentenceSpan[];
	fits(span: SentenceSpan): boolean;
	// Exclusive end index of the chunk that starts at spans[cursor]
	chunkEnd(spans: SentenceSpan[], cursor: number): number;
	// Where the next chunk starts, carrying back the configured overlap
	nextStart(spans: SentenceSpan[], cursor: number, end: number): number;
};

function splitRangeByLength(
	text: string,
	start: number,
	end: number,
	limit: number
): SentenceSpan[] {
	const spans: SentenceSpan[] = [];
	let cursor = start;

	while (cursor < end) {
		while (cursor < end && /\s/.test(text[cursor])) {
			cursor += 1;
		}

		if (cursor >= end) break;

		let splitAt = Math.min(cursor + limit, end);

		if (splitAt < end) {
			const window = text.slice(cursor, splitAt + 1);
			const boundary = Math.max(window.lastIndexOf('\n'), window.lastIndexOf(' '));

			// Prefer a natural boundary, but hard-split unbroken text so the size limit is guaranteed.
			if (boundary > 0) {
				splitAt = cursor + boundary;
			}
		}

		const value = normalizeWhitespace(text.slice(cursor, splitAt));

		if (value) {
			spans.push({
				text: value,
				start: cursor,
				end: splitAt
			});
		}

		cursor = splitAt;
	}

	return spans;
}

export function characterBudget(maxCharacters: number, overlapSentences: number): ChunkBudget {
	return {
		splitRange: (text, start, end) => splitRangeByLength(text, start, end, maxCharacters),
		fits: (span) => span.end - span.start <= maxCharacters,
		chunkEnd(spans, cursor) {
			let end = cursor + 1;

			while (end < spans.length) {
				const candidateLength = spans[end].end - spans[cursor].start;

				if (candidateLength > maxCharacters) {
					break;
				}

				end += 1;
			}

			return end;
		},
		// Keep a small overlap so answers split across chunk boundaries retain context
		nextStart: (_spans, cursor, end) => Math.max(end - overlapSentences, cursor + 1)
	};
}

function splitRangeByTokens(
	text: string,
	start: number,
	end: number,
	maxTokens: number,
	countTokens: (text: string) => number
): SentenceSpan[] {
	const spans: SentenceSpan[] = [];
	let spanStart = -1;
	let spanEnd = start;
	let total = 0;

	const push = () => {
		const value = normalizeWhitespace(text.slice(spanStart, spanEnd));
		if (value) spans.push({ text: value, start: spanStart, end: spanEnd });
	};

	// Split at word boundaries; a single word longer than the budget stays whole
	for (const word of text.slice(start, end).matchAll(/\S+/g)) {
		const wordStart = start + (word.index ?? 0);
		const tokens = countTokens(word[0]);

		if (spanStart >= 0 && total + tokens > maxTokens) {
			push();
			spanStart = -1;
			total = 0;
		}

		if (spanStart < 0) spanStart = wordStart;
		spanEnd = wordStart + word[0].length;
		total += tokens;
	}

	if (spanStart >= 0) push();
	return spans;
}

export function tokenBudget(
	maxTokens: number,
	overlapTokens: number,
	countTokens: (text: string) => number
): ChunkBudget {
	const counts = new WeakMap<SentenceSpan, number>();
	const tokensOf = (span: SentenceSpan) => {
		let count = counts.get(span);
		if (count === undefined) {
			count = countTokens(span.text);
			counts.set(span, count);
		}
		return count;
	};

	return {
		splitRange: (text, start, end) => splitRangeByTokens(text, start, end, maxTokens, countTokens),
		fits: (span) => tokensOf(span) <= maxTokens,
		chunkEnd(spans, cursor) {
			let total = tokensOf(spans[cursor]);
			let end = cursor + 1;

			while (end < spans.length && total + tokensOf(spans[end]) <= maxTokens) {
				total += tokensOf(spans[end]);
				end += 1;
			}

			return end;
		},
		nextStart(spans, cursor, end) {
			if (overlapTokens === 0 || end >= spans.length) return end;

			// Carry back whole sentences: always the last one, then more while they fit the overlap,
			// but never so many that the next sentence no longer fits beside them.
			const room = maxTokens - tokensOf(spans[end]);
			let start = end;
			let carried = 0;

			while (start - 1 > cursor) {
				const tokens = tokensOf(spans[start - 1]);
				const withinOverlap = start === end || carried + tokens <= overlapTokens;
				if (!withinOverlap || carried + tokens > room) break;
				carried += tokens;
				start -= 1;
			}

			return start;
		}
	};
}

// Token budgets count with the embedding model's tokenizer, so they load it first
export async function loadChunkBudget(): Promise<ChunkBudget> {
	if (chunkingConfig.unit === 'characters') {
		return characterBudget(chunkingConfig.maxCharacters, chunkingConfig.overlapSentences);
	}

	return tokenBudget(
		chunkingConfig.maxTokens,
		chunkingConfig.overlapTokens,
		await getEmbeddingTokenCounter()
	);
}

function splitOversizedSpans(
	text: string,
	spans: SentenceSpan[],
	budget: ChunkBudget
): SentenceSpan[] {
	return spans.flatMap((span) => {
		if (budget.fits(span)) {
			return span;
		}

		return budget.splitRange(text, span.start, span.end);
	});
}

export function cleanPageText(text: string): string {
	// Keep page cleanup light as extraction does most de-noising
	return text
		.replace(/\r\n/g, '\n')
		.split('\n')
		.map((line) => normalizeWhitespace(line))
		.filter(Boolean)
		.join('\n');
}

// Prevents "Dr. Evil" --> [Dr, Evil]
const PROTECTED_ABBREVIATIONS = new Set([
	'dr.',
	'mr.',
	'mrs.',
	'ms.',
	'prof.',
	'lt.',
	'col.',
	'capt.',
	'maj.',
	'gen.',
	'sgt.',
	'cpl.',
	'pfc.',
	'spc.'
]);

function shouldSplitAtPeriod(text: string, index: number): boolean {
	const prevChar = text[index - 1] ?? '';
	const nextChar = text[index + 1] ?? '';

	// Decimal values like "3.6" should stay in one span
	if (/\d/.test(prevChar) && /\d/.test(nextChar)) {
		return false;
	}

	// Check the token before this period for common abbreviations like "Dr."
	const tokenStart =
		Math.max(text.lastIndexOf(' ', index - 1), text.lastIndexOf('\n', index - 1)) + 1;
	const token = text.slice(tokenStart, index + 1).toLowerCase();

	if (PROTECTED_ABBREVIATIONS.has(token)) {
		return false;
	}

	// Avoid splitting after initialisms like "U.S.A."
	if (/[A-Z](?:\.[A-Z])+\.$/.test(text.slice(Math.max(0, index - 12), index + 1))) {
		return false;
	}

	return true;
}

export function splitSentencesWithOffsets(text: string): SentenceSpan[] {
	const normalized = normalizeWhitespace(text);
	if (!normalized) return [];

	const spans: SentenceSpan[] = [];
	let sentenceStart = 0;

	// Split only on sentence punctuation followed by whitespace/end so inline punctuation stays intact
	for (let index = 0; index < normalized.length; index += 1) {
		const char = normalized[index];
		const nextChar = normalized[index + 1] ?? '';
		const hasSentenceBreak = nextChar === '' || /\s/.test(nextChar);
		const isBoundary =
			((char === '!' || char === '?') && hasSentenceBreak) ||
			(char === '.' && hasSentenceBreak && shouldSplitAtPeriod(normalized, index));

		if (!isBoundary) continue;

		const value = normalized.slice(sentenceStart, index + 1).trim();

		if (value) {
			spans.push({
				text: value,
				start: sentenceStart,
				end: index + 1
			});
		}

		sentenceStart = index + 1;
	}

	const tail = normalized.slice(sentenceStart).trim();

	if (tail) {
		// PDF text may lack final punctuation, so keep the remaining tail
		spans.push({
			text: tail,
			start: sentenceStart,
			end: normalized.length
		});
	}

	return spans;
}

function getChunkContent(content: string, startChar: number, endChar: number): string {
	return normalizeWhitespace(content.slice(startChar, endChar).replace(/\n/g, ' '));
}

function chunkSentenceSpans(
	page: ExtractedChunk,
	content: string,
	spans: SentenceSpan[],
	budget: ChunkBudget
): ParsedChunk[] {
	const boundedSpans = splitOversizedSpans(content, spans, budget);
	if (boundedSpans.length === 0) return [];

	const chunks: ParsedChunk[] = [];
	let chunkIndex = 0;
	let cursor = 0;

	while (cursor < boundedSpans.length) {
		const end = budget.chunkEnd(boundedSpans, cursor);
		const selected = boundedSpans.slice(cursor, end);
		const startChar = selected[0].start;
		const endChar = selected[selected.length - 1].end;
		const chunkContent = getChunkContent(content, startChar, endChar);

		if (chunkContent && countWords(chunkContent) >= MIN_WORDS) {
			chunks.push({
				chunkId: buildChunkId(page, chunkIndex, chunkContent),
				chunkType: page.chunkType,
				source: page.source,
				pageIndex: page.pageIndex,
				chunkIndex,
				content: chunkContent,
				startChar,
				endChar
			});
			chunkIndex += 1;
		}

		if (end >= boundedSpans.length) {
			break;
		}

		cursor = budget.nextStart(boundedSpans, cursor, end);
	}

	return chunks;
}

function preparePageContent(page: ExtractedChunk): string {
	switch (page.chunkType) {
		case 'TEXT':
			return cleanPageText(page.content);
		default:
			return normalizeWhitespace(page.content);
	}
}

function chunkPage(page: ExtractedChunk, budget: ChunkBudget): ParsedChunk[] {
	const content = preparePageContent(page);
	if (!content) return [];

	if (page.chunkType === 'IMAGE') {
		return [
			{
				chunkId: buildChunkId(page, 0, content),
				chunkType: page.chunkType,
				source: page.source,
				pageIndex: page.pageIndex,
				chunkIndex: 0,
				content
			}
		];
	}

	if (page.chunkType === 'TABLE') {
		// Tables may contain many rows, so keep enforcing the embedding size limit for them.
		return budget.splitRange(content, 0, content.length).map((span, chunkIndex) => {
			const chunkContent = span.text;

			return {
				chunkId: buildChunkId(page, chunkIndex, chunkContent),
				chunkType: page.chunkType,
				source: page.source,
				pageIndex: page.pageIndex,
				chunkIndex,
				content: chunkContent
			};
		});
	}

	const spans = splitSentencesWithOffsets(content);
	return chunkSentenceSpans(page, content, spans, budget);
}

export function chunkPages(pages: ExtractedChunk[], budget: ChunkBudget): ParsedChunk[] {
	return pages.flatMap((page) => chunkPage(page, budget));
}
