import { API_HIPPORAG } from '$lib/constants';
import type {
	ApiDocumentIngestProgress,
	ApiHippoRagBuildEvent,
	ApiHippoRagIndexStatus
} from '$lib/types';
import { apiFetch, apiStream, parseNdjsonStream } from '$lib/utils';

export class HippoRagService {
	static status() {
		return apiFetch<ApiHippoRagIndexStatus>(API_HIPPORAG);
	}

	static async build(
		rebuild: boolean,
		onProgress?: (progress: ApiDocumentIngestProgress) => void,
		signal?: AbortSignal
	): Promise<ApiHippoRagIndexStatus> {
		const response = await apiStream(API_HIPPORAG, {
			method: 'POST',
			body: JSON.stringify({ rebuild }),
			signal
		});

		for await (const event of parseNdjsonStream<ApiHippoRagBuildEvent>(response, signal)) {
			if (event.status === 'progress') onProgress?.(event);
			else if (event.status === 'complete') return event.result;
			else throw new Error(event.message);
		}
		throw new Error('HippoRAG2 index build ended before completion.');
	}
}
