import type { SearchChunkType, SearchMatchBase } from './search-shared';
import { searchWithRetrievalMode } from './retrieval-registry';
import { DEFAULT_ASSISTANT_CONFIG, RAG_CHUNK_CHARACTER_LIMIT } from '$lib/constants';
import { RetrievalMode } from '$lib/enums';
import { compactText } from '$lib/server/utils/values';

const MAX_PREVIEW_CHARS = 200;
const ENV_RETRIEVAL_MODES: readonly RetrievalMode[] = Object.values(RetrievalMode);
const DEFAULT_RETRIEVAL_MODE =
	ENV_RETRIEVAL_MODES.find((mode) => mode === process.env.RAG_RETRIEVAL_MODE) ??
	DEFAULT_ASSISTANT_CONFIG.retrievalMode;

export type RagRetrievalMode = RetrievalMode;

export type RagSource = {
	title: string;
	description: string;
	documentId: string;
	chunkId: string;
	sourceType: SearchMatchBase['sourceType'];
	pageIndex: number;
	chunkIndex: number;
};

export type RagContextResult = {
	mode: RagRetrievalMode;
	contextBlock: string;
	sources: RagSource[];
};

// Format retrieved chunks in the old RAG prompt style
function formatContext(matches: SearchMatchBase[]) {
	if (matches.length === 0) return '';

	const items = matches.map((match) => {
		const content = compactText(match.content, RAG_CHUNK_CHARACTER_LIMIT);
		const source = match.sourceTitle || match.sourcePath || 'unknown';

		return `- ${content} (source: ${source})`;
	});

	return ['Relevant context:', ...items].join('\n');
}

// Sources are the user-facing citation list, so keep them shorter than the model context
export function buildSources(matches: SearchMatchBase[]): RagSource[] {
	return matches.map((match) => {
		const preview = compactText(match.content, MAX_PREVIEW_CHARS);

		return {
			title: match.sourceTitle,
			// Transcripts have no pages, so only paged sources get a page reference
			description:
				match.sourceType === 'AUDIO' ? preview : `Page ${match.pageIndex + 1}: ${preview}`,
			documentId: match.documentId,
			chunkId: match.chunkId,
			sourceType: match.sourceType,
			pageIndex: match.pageIndex,
			chunkIndex: match.chunkIndex
		};
	});
}

// Chat uses the profile selection by default. RAG_RETRIEVAL_MODE can override it for local testing.
export async function retrieveRagContext({
	question,
	documentIds = [],
	chunkTypes = ['TEXT', 'TABLE', 'IMAGE'],
	topK = DEFAULT_ASSISTANT_CONFIG.ragTopK,
	mode = DEFAULT_RETRIEVAL_MODE
}: {
	question: string;
	documentIds?: string[];
	chunkTypes?: SearchChunkType[];
	topK?: number;
	mode?: RagRetrievalMode;
}): Promise<RagContextResult> {
	const searchOptions = {
		query: question,
		topK,
		documentIds,
		chunkTypes
	};
	const matches = await searchWithRetrievalMode(mode, searchOptions);

	return {
		mode,
		contextBlock: formatContext(matches),
		sources: buildSources(matches)
	};
}
