import { getExperimentalRetrievalTrainingEnabled } from '$lib/server/database/app-state';

export class RetrievalTrainingDisabledError extends Error {
	constructor() {
		super('User rating training is disabled in Settings → Agent → Experimental.');
		this.name = 'RetrievalTrainingDisabledError';
	}
}

export async function requireExperimentalRetrievalTraining(): Promise<void> {
	if (!(await getExperimentalRetrievalTrainingEnabled()))
		throw new RetrievalTrainingDisabledError();
}
