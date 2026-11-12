import { setStructuredPdfParsing } from '#lib/server/database/app-state';
import { getDocumentParsingStatus } from '#lib/server/documents/document-parsing';
import { isLayoutModelInstalled } from '#lib/server/rag/chunk/layout/heron-model';
import type { ApiDocumentParsingUpdate } from '#lib/types';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => Response.json(await getDocumentParsingStatus());

export const PATCH: RequestHandler = async ({ request }) => {
	const body = (await request.json().catch(() => null)) as Partial<ApiDocumentParsingUpdate> | null;
	if (typeof body?.structuredPdfParsing !== 'boolean') {
		return Response.json({ error: 'structuredPdfParsing must be a boolean.' }, { status: 400 });
	}
	if (body.structuredPdfParsing && !isLayoutModelInstalled()) {
		return Response.json(
			{ error: 'Download the layout model before turning on structure-aware parsing.' },
			{ status: 409 }
		);
	}

	await setStructuredPdfParsing(body.structuredPdfParsing);
	return Response.json(await getDocumentParsingStatus());
};
