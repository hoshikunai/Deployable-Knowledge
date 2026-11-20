function readBoolean(name: string): boolean {
	const raw = process.env[name]?.trim();
	if (!raw || raw === 'false') return false;
	if (raw === 'true') return true;
	throw new Error(`${name} must be "true" or "false", received "${raw}".`);
}

// Removing repeated page-edge lines can remove real text, so cleanup is opt-in.
export const pdfPageCleanup = readBoolean('RAG_PDF_PAGE_CLEANUP');
export const lowMemoryPdfPrepare = readBoolean('RAG_LOW_MEMORY_PDF_PREPARE');
