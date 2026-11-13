import { installTableModels } from '#lib/server/rag/chunk/tables/table-models';
import { modelDownloadResponse } from '#lib/server/utils/model-download-response';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async () =>
	modelDownloadResponse('Table model download', (onProgress) => installTableModels(onProgress));
