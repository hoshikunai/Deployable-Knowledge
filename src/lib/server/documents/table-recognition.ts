import type { ApiTableRecognitionStatus } from '#lib/types';
import { createLogger } from '#lib/server/logging';
import { getTableRecognition } from '#lib/server/database/app-state';
import { pdfPageCleanup } from '#lib/server/rag/chunk/pdf-config';
import {
	areTableModelsInstalled,
	TABLE_MODELS_SIZE_BYTES
} from '#lib/server/rag/chunk/tables/table-models';

const log = createLogger('documents');

export async function getTableRecognitionStatus(): Promise<ApiTableRecognitionStatus> {
	return {
		enabled: await getTableRecognition(),
		modelsInstalled: areTableModelsInstalled(),
		modelsSizeBytes: TABLE_MODELS_SIZE_BYTES,
		pdfPageCleanup
	};
}

export async function shouldRecognizeTables(): Promise<boolean> {
	if (!(await getTableRecognition())) return false;
	if (areTableModelsInstalled()) return true;
	log.warn('Table recognition is on but its models are missing; using pdf-parse tables');
	return false;
}
