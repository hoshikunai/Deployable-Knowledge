import { DocumentParsingService } from '#lib/services';
import type { ApiDocumentIngestProgress, ApiDocumentParsingStatus } from '#lib/types';

function errorMessage(error: unknown, fallback: string): string {
	return error instanceof Error ? error.message : fallback;
}

class DocumentParsingStore {
	status = $state<ApiDocumentParsingStatus | null>(null);
	saving = $state(false);
	downloading = $state(false);
	progress = $state<ApiDocumentIngestProgress | null>(null);
	error = $state<string | null>(null);

	async refresh(): Promise<void> {
		try {
			this.status = await DocumentParsingService.getStatus();
		} catch (error) {
			this.error = errorMessage(error, 'Could not load the document parsing settings.');
		}
	}

	async downloadModel(): Promise<boolean> {
		if (this.downloading) return false;

		this.downloading = true;
		this.error = null;
		this.progress = { percent: 0, label: 'Downloading layout model', message: 'Starting.' };
		try {
			await DocumentParsingService.downloadModel((progress) => (this.progress = progress));
			await this.refresh();
			return true;
		} catch (error) {
			this.error = errorMessage(error, 'Layout model download failed.');
			return false;
		} finally {
			this.downloading = false;
			this.progress = null;
		}
	}

	async setStructuredPdfParsing(enabled: boolean): Promise<void> {
		if (!this.status || this.saving || this.downloading) return;
		if (this.status.structuredPdfParsing === enabled) return;
		if (enabled && !this.status.modelInstalled && !(await this.downloadModel())) return;

		this.saving = true;
		this.error = null;
		try {
			this.status = await DocumentParsingService.update({ structuredPdfParsing: enabled });
		} catch (error) {
			this.error = errorMessage(error, 'Could not change the document parsing setting.');
		} finally {
			this.saving = false;
		}
	}
}

export const documentParsingStore = new DocumentParsingStore();
