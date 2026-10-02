// Optional document ordering for hybrid results. 'sum-top2' scores each document by the sum
// of its two best chunk scores, so documents relevant in more than one place rank higher,
// then returns documents in that order with each document's chunks in score order.

import type { ScoredSearchMatch } from './search-shared';

export type DocumentOrder = 'none' | 'sum-top2';

export const DOCUMENT_ORDERS: readonly DocumentOrder[] = ['none', 'sum-top2'];

const CHUNKS_PER_DOCUMENT_SCORE = 2;

// Expects matches in descending score order; returns the first topK after regrouping
export function orderByDocument(
	matches: readonly ScoredSearchMatch[],
	order: DocumentOrder,
	topK: number
): ScoredSearchMatch[] {
	if (order === 'none') return matches.slice(0, topK);

	const byDocument = new Map<string, ScoredSearchMatch[]>();
	for (const match of matches) {
		const chunks = byDocument.get(match.documentId) ?? [];
		chunks.push(match);
		byDocument.set(match.documentId, chunks);
	}

	const documentScore = (chunks: ScoredSearchMatch[]) =>
		chunks.slice(0, CHUNKS_PER_DOCUMENT_SCORE).reduce((total, chunk) => total + chunk.score, 0);

	// Stable sort keeps the incoming order for documents with equal scores
	return [...byDocument.values()]
		.sort((left, right) => documentScore(right) - documentScore(left))
		.flat()
		.slice(0, topK);
}
