import { setTableRecognition } from '#lib/server/database/app-state';
import { getTableRecognitionStatus } from '#lib/server/documents/table-recognition';
import { areTableModelsInstalled } from '#lib/server/rag/chunk/tables/table-models';
import type { ApiTableRecognitionUpdate } from '#lib/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => Response.json(await getTableRecognitionStatus());

export const PATCH: RequestHandler = async ({ request }) => {
	const body = (await request
		.json()
		.catch(() => null)) as Partial<ApiTableRecognitionUpdate> | null;
	if (typeof body?.enabled !== 'boolean') {
		return Response.json({ error: 'enabled must be a boolean.' }, { status: 400 });
	}
	if (body.enabled && !areTableModelsInstalled()) {
		return Response.json(
			{ error: 'Download the table models before turning on table recognition.' },
			{ status: 409 }
		);
	}

	await setTableRecognition(body.enabled);
	return Response.json(await getTableRecognitionStatus());
};
