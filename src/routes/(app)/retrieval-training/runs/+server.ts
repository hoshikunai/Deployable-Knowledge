import { json, type RequestHandler } from '@sveltejs/kit';
import { HUMAN_EXPERT_FEEDBACK_SOURCE, RETRIEVAL_FEEDBACK_SOURCES } from '$lib/constants';
import {
	RetrievalTrainingReadinessError,
	runInitialRetrievalTraining
} from '$lib/server/rag/training';
import type {
	ApiRetrievalTrainingRunRequest,
	ApiRetrievalTrainingRunResponse,
	RetrievalFeedbackSource
} from '$lib/types';

function isRequestObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readFeedbackSource(request: Request): Promise<RetrievalFeedbackSource | null> {
	const rawBody = await request.text();
	if (!rawBody.trim()) return HUMAN_EXPERT_FEEDBACK_SOURCE;

	let value: unknown;
	try {
		value = JSON.parse(rawBody);
	} catch {
		return null;
	}

	if (!isRequestObject(value)) return null;

	const input: ApiRetrievalTrainingRunRequest = {
		feedbackSource:
			typeof value.feedbackSource === 'string' &&
			RETRIEVAL_FEEDBACK_SOURCES.includes(value.feedbackSource as RetrievalFeedbackSource)
				? (value.feedbackSource as RetrievalFeedbackSource)
				: undefined
	};

	if (value.feedbackSource !== undefined && input.feedbackSource === undefined) return null;
	return input.feedbackSource ?? HUMAN_EXPERT_FEEDBACK_SOURCE;
}

export const POST: RequestHandler = async ({ request }) => {
	try {
		const feedbackSource = await readFeedbackSource(request);
		if (!feedbackSource) {
			return json({ error: 'Feedback source must be human_expert or ai_proxy.' }, { status: 400 });
		}

		const result = await runInitialRetrievalTraining(feedbackSource);

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
