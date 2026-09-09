import { json, type RequestHandler } from '@sveltejs/kit';
import { HUMAN_EXPERT_FEEDBACK_SOURCE } from '$lib/constants';
import {
	RetrievalTrainingReadinessError,
	runInitialRetrievalTraining
} from '$lib/server/rag/training';
import type { ApiRetrievalTrainingRunResponse } from '$lib/types';
import {
	requireExperimentalRetrievalTraining,
	RetrievalTrainingDisabledError
} from '$lib/server/retrieval/experimental-retrieval-training';

function isRequestObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readTrainingRequest(request: Request): Promise<boolean> {
	const rawBody = await request.text();
	if (!rawBody.trim()) return true;

	let value: unknown;
	try {
		value = JSON.parse(rawBody);
	} catch {
		return false;
	}

	if (!isRequestObject(value)) return false;
	return (
		value.feedbackSource === undefined || value.feedbackSource === HUMAN_EXPERT_FEEDBACK_SOURCE
	);
}

export const POST: RequestHandler = async ({ request }) => {
	try {
		await requireExperimentalRetrievalTraining();
		if (!(await readTrainingRequest(request)))
			return json({ error: 'Only human_expert training is supported.' }, { status: 400 });
		const result = await runInitialRetrievalTraining();

		return json({
			runId: result.runId,
			modelId: result.modelId,
			feedbackSource: result.feedbackSource,
			trainingExamples: result.trainingExamples,
			validationExamples: result.validationExamples,
			trainingPairs: result.trainingPairs,
			validationPairs: result.validationPairs,
			distinctQueries: result.distinctQueries,
			evaluation: result.evaluation
		} satisfies ApiRetrievalTrainingRunResponse);
	} catch (error) {
		if (error instanceof RetrievalTrainingDisabledError)
			return json({ error: error.message }, { status: 403 });
		if (error instanceof RetrievalTrainingReadinessError) {
			return json(
				{
					error: error.message
				},
				{ status: 400 }
			);
		}

		console.error('[Retrieval Training] Local training failed.', error);

		return json(
			{
				error: 'Local retrieval training failed.'
			},
			{ status: 500 }
		);
	}
};
