import { json, type RequestHandler } from '@sveltejs/kit';
import { getActiveRetrievalModelId } from '$lib/server/database/app-state';
import {
	activateRetrievalModel,
	deactivateRetrievalModel
} from '$lib/server/rag/search/active-retrieval-model';
import type {
	ApiRetrievalModelActivationRequest,
	ApiRetrievalModelActivationResponse
} from '$lib/types';
import {
	requireExperimentalRetrievalTraining,
	RetrievalTrainingDisabledError
} from '$lib/server/retrieval/experimental-retrieval-training';

function isRequestObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export const GET: RequestHandler = async () => {
	return json({
		activeModelId: await getActiveRetrievalModelId()
	} satisfies ApiRetrievalModelActivationResponse);
};

export const PUT: RequestHandler = async ({ request }) => {
	try {
		await requireExperimentalRetrievalTraining();
	} catch (error) {
		if (error instanceof RetrievalTrainingDisabledError)
			return json({ error: error.message }, { status: 403 });
		throw error;
	}
	let value: unknown;

	try {
		value = await request.json();
	} catch {
		return json({ error: 'A JSON request body is required.' }, { status: 400 });
	}

	if (!isRequestObject(value)) {
		return json({ error: 'A JSON request body is required.' }, { status: 400 });
	}

	const modelId = typeof value.modelId === 'string' ? value.modelId.trim() : '';
	if (!modelId) {
		return json({ error: 'A model id is required.' }, { status: 400 });
	}

	try {
		const input: ApiRetrievalModelActivationRequest = { modelId };
		const model = await activateRetrievalModel(input.modelId);

		return json({
			activeModelId: model.id
		} satisfies ApiRetrievalModelActivationResponse);
	} catch (error) {
		const message =
			error instanceof Error ? error.message : 'The retrieval model could not be activated.';
		return json({ error: message }, { status: 400 });
	}
};

export const DELETE: RequestHandler = async () => {
	try {
		await requireExperimentalRetrievalTraining();
	} catch (error) {
		if (error instanceof RetrievalTrainingDisabledError)
			return json({ error: error.message }, { status: 403 });
		throw error;
	}
	await deactivateRetrievalModel();

	return json({
		activeModelId: null
	} satisfies ApiRetrievalModelActivationResponse);
};
