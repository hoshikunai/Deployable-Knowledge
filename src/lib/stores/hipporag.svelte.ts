import { HippoRagService } from '$lib/services';
import type { ApiDocumentIngestProgress, ApiHippoRagIndexStatus } from '$lib/types';

class HippoRagStore {
	status = $state<ApiHippoRagIndexStatus | null>(null);
	progress = $state<ApiDocumentIngestProgress | null>(null);
	loading = $state(false);
	building = $state(false);
	error = $state<string | null>(null);

	async load(): Promise<void> {
		if (this.loading) return;
		this.loading = true;
		this.error = null;
		try {
			this.status = await HippoRagService.status();
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		} finally {
			this.loading = false;
		}
	}

	async build(rebuild: boolean): Promise<void> {
		if (this.building) return;
		this.building = true;
		this.error = null;
		this.progress = {
			percent: 0,
			label: 'Building HippoRAG2 index',
			message: 'Preparing index'
		};
		try {
			this.status = await HippoRagService.build(rebuild, (progress) => (this.progress = progress));
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
			throw error;
		} finally {
			this.building = false;
			this.progress = null;
			await this.load();
		}
	}
}

export const hippoRagStore = new HippoRagStore();
