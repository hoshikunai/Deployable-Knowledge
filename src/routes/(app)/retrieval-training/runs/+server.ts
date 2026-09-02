import { json, type RequestHandler } from '@sveltejs/kit';
import {
	RetrievalTrainingReadinessError,
	runInitialRetrievalTraining
} from '$lib/server/rag/training';
import type { ApiRetrievalTrainingRunResponse } from '$lib/types';

export const POST: RequestHandler = async () => {
	try {
		const result = await runInitialRetrievalTraining();

		return json({
			runId: result.runId,
			modelId: result.modelId,
			trainingExamples: result.trainingExamples,
			validationExamples: result.validationExamples,
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
