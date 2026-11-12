import { installLayoutModel } from '#lib/server/rag/chunk/layout/heron-model';
import { modelDownloadResponse } from '#lib/server/utils/model-download-response';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async () =>
	modelDownloadResponse('Layout model download', (onProgress) => installLayoutModel(onProgress));
