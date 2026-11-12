import type { ApiDocumentParsingStatus } from '#lib/types';
import { createLogger } from '#lib/server/logging';
import { getStructuredPdfParsing } from '#lib/server/database/app-state';
import {
	isLayoutModelInstalled,
	LAYOUT_MODEL_SIZE_BYTES
} from '#lib/server/rag/chunk/layout/heron-model';

const log = createLogger('documents');

export async function getDocumentParsingStatus(): Promise<ApiDocumentParsingStatus> {
	return {
		structuredPdfParsing: await getStructuredPdfParsing(),
		modelInstalled: isLayoutModelInstalled(),
		modelSizeBytes: LAYOUT_MODEL_SIZE_BYTES
	};
}

export async function shouldUseStructuredParsing(): Promise<boolean> {
	if (!(await getStructuredPdfParsing())) return false;
	if (isLayoutModelInstalled()) return true;
	log.warn(
		'Structure-aware PDF parsing is on but the layout model is missing; using plain extraction'
	);
	return false;
}
