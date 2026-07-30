import { API_SEARCH } from '$lib/constants';
import type { RetrievalMode } from '$lib/enums';
import type { ApiSearchMatch } from '$lib/types';
import { apiFetch } from '$lib/utils';

export class SearchService {
	static search(
		query: string,
		topK: number,
		mode: RetrievalMode,
		documentIds: readonly string[] = []
	) {
		const params = new URLSearchParams({ query, topK: String(topK), mode });
		for (const id of documentIds) params.append('documentIds', id);
		return apiFetch<ApiSearchMatch[]>(`${API_SEARCH}?${params}`);
	}
}
