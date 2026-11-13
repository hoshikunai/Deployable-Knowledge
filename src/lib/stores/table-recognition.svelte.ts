import { TableRecognitionService } from '#lib/services';
import type { ApiDocumentIngestProgress, ApiTableRecognitionStatus } from '#lib/types';

function errorMessage(error: unknown, fallback: string): string {
	return error instanceof Error ? error.message : fallback;
}

class TableRecognitionStore {
	status = $state<ApiTableRecognitionStatus | null>(null);
	saving = $state(false);
	downloading = $state(false);
	progress = $state<ApiDocumentIngestProgress | null>(null);
	error = $state<string | null>(null);

	async refresh(): Promise<void> {
		try {
			this.status = await TableRecognitionService.getStatus();
		} catch (error) {
			this.error = errorMessage(error, 'Could not load the table recognition setting.');
		}
	}

	async downloadModels(): Promise<boolean> {
		if (this.downloading) return false;

		this.downloading = true;
		this.error = null;
		this.progress = { percent: 0, label: 'Downloading table models', message: 'Starting.' };
		try {
			await TableRecognitionService.downloadModels((progress) => (this.progress = progress));
			await this.refresh();
			return true;
		} catch (error) {
			this.error = errorMessage(error, 'Table model download failed.');
			return false;
		} finally {
			this.downloading = false;
			this.progress = null;
		}
	}

	async setEnabled(enabled: boolean): Promise<void> {
		if (!this.status || this.saving || this.downloading) return;
		if (this.status.enabled === enabled) return;
		if (enabled && !this.status.modelsInstalled && !(await this.downloadModels())) return;

		this.saving = true;
		this.error = null;
		try {
			this.status = await TableRecognitionService.update({ enabled });
		} catch (error) {
			this.error = errorMessage(error, 'Could not change the table recognition setting.');
		} finally {
			this.saving = false;
		}
	}
}

export const tableRecognitionStore = new TableRecognitionStore();
