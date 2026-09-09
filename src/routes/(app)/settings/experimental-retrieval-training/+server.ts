import { json, type RequestHandler } from '@sveltejs/kit';
import {
	getExperimentalRetrievalTrainingEnabled,
	setExperimentalRetrievalTrainingEnabled
} from '$lib/server/database/app-state';
import type { ExperimentalRetrievalTrainingSettings } from '$lib/types';

export const GET: RequestHandler = async () =>
	json({
		enabled: await getExperimentalRetrievalTrainingEnabled()
	} satisfies ExperimentalRetrievalTrainingSettings);

export const PATCH: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as Partial<ExperimentalRetrievalTrainingSettings>;
	if (typeof body.enabled !== 'boolean')
		return json({ error: 'enabled must be a boolean.' }, { status: 400 });
	return json({
		enabled: await setExperimentalRetrievalTrainingEnabled(body.enabled)
	} satisfies ExperimentalRetrievalTrainingSettings);
};
