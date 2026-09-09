import { API_EXPERIMENTAL_RETRIEVAL_TRAINING } from '$lib/constants';
import type { ExperimentalRetrievalTrainingSettings } from '$lib/types';
import { apiFetch, apiPatch } from '$lib/utils';

export class ExperimentalRetrievalTrainingService {
	static get() {
		return apiFetch<ExperimentalRetrievalTrainingSettings>(API_EXPERIMENTAL_RETRIEVAL_TRAINING);
	}

	static update(enabled: boolean) {
		return apiPatch<ExperimentalRetrievalTrainingSettings, ExperimentalRetrievalTrainingSettings>(
			API_EXPERIMENTAL_RETRIEVAL_TRAINING,
			{ enabled }
		);
	}
}
