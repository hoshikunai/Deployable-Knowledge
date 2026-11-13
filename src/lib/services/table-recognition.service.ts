import { API_TABLE_RECOGNITION } from '#lib/constants';
import type {
	ApiDocumentIngestProgress,
	ApiTableRecognitionStatus,
	ApiTableRecognitionUpdate
} from '#lib/types';
import { apiFetch, apiPatch } from '#lib/utils';
import { streamModelDownload } from './model-download';

export class TableRecognitionService {
	static getStatus() {
		return apiFetch<ApiTableRecognitionStatus>(API_TABLE_RECOGNITION.BASE);
	}

	static update(update: ApiTableRecognitionUpdate) {
		return apiPatch<ApiTableRecognitionStatus, ApiTableRecognitionUpdate>(
			API_TABLE_RECOGNITION.BASE,
			update
		);
	}

	static downloadModels(onProgress: (progress: ApiDocumentIngestProgress) => void): Promise<void> {
		return streamModelDownload(
			API_TABLE_RECOGNITION.MODELS,
			null,
			'Downloading table models',
			onProgress
		);
	}
}
