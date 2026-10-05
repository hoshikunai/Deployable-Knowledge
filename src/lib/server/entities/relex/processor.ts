import type { PreTrainedTokenizer } from '@huggingface/transformers';

export interface RelexModelConfig {
	maxWidth: number;
	maxWords: number;
	entityToken: string;
	relationToken: string;
	separatorToken: string;
}

export interface RelexWord {
	text: string;
	start: number;
	end: number;
}

export interface RelexSpecialTokens {
	cls: number;
	sep: number;
}

export interface RelexInputs {
	words: RelexWord[];
	tokenCount: number;
	spanCount: number;
	inputIds: BigInt64Array;
	attentionMask: BigInt64Array;
	wordsMask: BigInt64Array;
	textLengths: BigInt64Array;
	spanIdx: BigInt64Array;
	spanMask: Uint8Array;
}

const WORD_PATTERN = /[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S/gu;

export function splitWords(text: string, maxWords: number): RelexWord[] {
	const words: RelexWord[] = [];
	for (const match of text.matchAll(WORD_PATTERN)) {
		if (words.length === maxWords) break;
		words.push({ text: match[0], start: match.index, end: match.index + match[0].length });
	}
	return words;
}

export function promptWords(
	config: RelexModelConfig,
	entityLabels: readonly string[],
	relationLabels: readonly string[]
): string[] {
	return [
		...entityLabels.flatMap((label) => [config.entityToken, label]),
		config.separatorToken,
		...relationLabels.flatMap((label) => [config.relationToken, label]),
		config.separatorToken
	];
}

function encodeTokens(
	tokenizer: PreTrainedTokenizer,
	special: RelexSpecialTokens,
	prompt: readonly string[],
	words: readonly RelexWord[]
): { ids: number[]; wordsMask: number[] } {
	const ids = [special.cls];
	const wordsMask = [0];
	let seenWords = 0;
	for (const word of [...prompt, ...words.map((item) => item.text)]) {
		const pieces = tokenizer.encode(word, { add_special_tokens: false });
		if (pieces.length > 0) seenWords += 1;
		pieces.forEach((id, piece) => {
			ids.push(id);
			wordsMask.push(piece === 0 && seenWords > prompt.length ? seenWords - prompt.length : 0);
		});
	}
	ids.push(special.sep);
	wordsMask.push(0);
	return { ids, wordsMask };
}

function buildSpans(
	wordCount: number,
	maxWidth: number
): { spanIdx: BigInt64Array; spanMask: Uint8Array } {
	const spanIdx = new BigInt64Array(wordCount * maxWidth * 2);
	const spanMask = new Uint8Array(wordCount * maxWidth);
	for (let start = 0; start < wordCount; start += 1) {
		for (let width = 0; width < maxWidth; width += 1) {
			const span = start * maxWidth + width;
			spanIdx[span * 2] = BigInt(start);
			spanIdx[span * 2 + 1] = BigInt(start + width);
			spanMask[span] = start + width < wordCount ? 1 : 0;
		}
	}
	return { spanIdx, spanMask };
}

export function encodeRelexInputs(
	tokenizer: PreTrainedTokenizer,
	special: RelexSpecialTokens,
	config: RelexModelConfig,
	words: RelexWord[],
	prompt: readonly string[]
): RelexInputs {
	const { ids, wordsMask } = encodeTokens(tokenizer, special, prompt, words);
	const { spanIdx, spanMask } = buildSpans(words.length, config.maxWidth);
	return {
		words,
		tokenCount: ids.length,
		spanCount: words.length * config.maxWidth,
		inputIds: BigInt64Array.from(ids, BigInt),
		attentionMask: new BigInt64Array(ids.length).fill(1n),
		wordsMask: BigInt64Array.from(wordsMask, BigInt),
		textLengths: BigInt64Array.of(BigInt(words.length)),
		spanIdx,
		spanMask
	};
}
