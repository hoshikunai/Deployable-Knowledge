import { RetrievalMode } from '$lib/enums';
import { searchHippoRag2 } from '../hipporag/search';
import { searchBm25 } from './bm25-search';
import { searchHybrid } from './hybrid-search';
import { searchSemantic } from './semantic-search';
import type { SearchMatchBase, SearchOptionsBase } from './search-shared';

type RetrievalSearch = (options: SearchOptionsBase) => Promise<SearchMatchBase[]>;

function withoutScore<T extends SearchMatchBase & { score: number }>({
	score: _score,
	...match
}: T): SearchMatchBase {
	return match;
}

const retrievalRegistry: Record<RetrievalMode, RetrievalSearch> = {
	[RetrievalMode.SEMANTIC]: async (options) =>
		(await searchSemantic(options)).results.map(withoutScore),
	[RetrievalMode.BM25]: async (options) => (await searchBm25(options)).results.map(withoutScore),
	[RetrievalMode.HYBRID]: async (options) => (await searchHybrid(options)).results,
	[RetrievalMode.HIPPORAG_2]: async (options) => (await searchHippoRag2(options)).results
};

export function searchWithRetrievalMode(
	mode: RetrievalMode,
	options: SearchOptionsBase
): Promise<SearchMatchBase[]> {
	return retrievalRegistry[mode](options);
}

export function isRetrievalMode(value: string): value is RetrievalMode {
	return Object.values(RetrievalMode).includes(value as RetrievalMode);
}
