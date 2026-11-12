import { API_DOCUMENT_PARSING } from '#lib/constants';
import type {
	ApiDocumentIngestProgress,
	ApiDocumentParsingStatus,
	ApiDocumentParsingUpdate
} from '#lib/types';
import { apiFetch, apiPatch } from '#lib/utils';
import { streamModelDownload } from './model-download';

export class DocumentParsingService {
	static getStatus() {
		return apiFetch<ApiDocumentParsingStatus>(API_DOCUMENT_PARSING.BASE);
	}

	static update(update: ApiDocumentParsingUpdate) {
		return apiPatch<ApiDocumentParsingStatus, ApiDocumentParsingUpdate>(
			API_DOCUMENT_PARSING.BASE,
			update
		);
	}

	static downloadModel(onProgress: (progress: ApiDocumentIngestProgress) => void): Promise<void> {
		return streamModelDownload(
			API_DOCUMENT_PARSING.MODEL,
			null,
			'Downloading layout model',
			onProgress
		);
	}
}
