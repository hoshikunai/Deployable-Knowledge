import type { RrfFusedMatch } from './rrf-fusion';

export type CandidateDedupMode = 'none' | 'exact' | 'near' | 'document-cap';

export const CANDIDATE_DEDUP_MODES: readonly CandidateDedupMode[] = [
	'none',
	'exact',
	'near',
	'document-cap'
];

const NEAR_DUPLICATE_JACCARD = 0.9;
const SHINGLE_WORDS = 5;

function normalizedWords(content: string): string[] {
	return content.toLowerCase().split(/\s+/).filter(Boolean);
}

function shingles(content: string): Set<string> {
	const words = normalizedWords(content);
	if (words.length <= SHINGLE_WORDS) return new Set([words.join(' ')]);

	const result = new Set<string>();
	for (let index = 0; index + SHINGLE_WORDS <= words.length; index += 1) {
		result.add(words.slice(index, index + SHINGLE_WORDS).join(' '));
	}
	return result;
}

function jaccard(left: Set<string>, right: Set<string>): number {
	const [smaller, larger] = left.size <= right.size ? [left, right] : [right, left];
	let shared = 0;
	for (const shingle of smaller) {
		if (larger.has(shingle)) shared += 1;
	}
	return shared / (left.size + right.size - shared);
}

export function dedupeCandidates(
	candidates: readonly RrfFusedMatch[],
	mode: CandidateDedupMode,
	limit: number,
	maxChunksPerDocument: number
): RrfFusedMatch[] {
	if (mode === 'none') return candidates.slice(0, limit);

	const kept: RrfFusedMatch[] = [];
	const seenTexts = new Set<string>();
	const keptShingles: Set<string>[] = [];
	const chunksPerDocument = new Map<string, number>();

	for (const candidate of candidates) {
		if (kept.length === limit) break;
		const { content, documentId } = candidate.match;

		if (mode === 'exact') {
			const key = normalizedWords(content).join(' ');
			if (seenTexts.has(key)) continue;
			seenTexts.add(key);
		} else if (mode === 'near') {
			const candidateShingles = shingles(content);
			const repeats = keptShingles.some(
				(existing) => jaccard(existing, candidateShingles) >= NEAR_DUPLICATE_JACCARD
			);
			if (repeats) continue;
			keptShingles.push(candidateShingles);
		} else {
			const count = chunksPerDocument.get(documentId) ?? 0;
			if (count >= maxChunksPerDocument) continue;
			chunksPerDocument.set(documentId, count + 1);
		}

		kept.push(candidate);
	}

	return kept;
}
