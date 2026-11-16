import type { ScoredSearchMatch } from './search-shared';

export type DocumentOrder = 'none' | 'sum-top2';

export const DOCUMENT_ORDERS: readonly DocumentOrder[] = ['none', 'sum-top2'];

const CHUNKS_PER_DOCUMENT_SCORE = 2;

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

	return [...byDocument.values()]
		.sort((left, right) => documentScore(right) - documentScore(left))
		.flat()
		.slice(0, topK);
}
