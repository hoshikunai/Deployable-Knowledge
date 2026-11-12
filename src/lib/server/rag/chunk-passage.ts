export const SECTION_PATH_SEPARATOR = ' > ';

type ChunkPassage = { content: string; sectionPath?: string | null };

/**
 * The text a chunk is embedded, reranked and quoted as: its content under the path of headings
 * it belongs to, so a passage that never repeats its section's subject still matches it.
 */
export function chunkPassageText({ content, sectionPath }: ChunkPassage): string {
	return sectionPath ? `${sectionPath}\n\n${content}` : content;
}
