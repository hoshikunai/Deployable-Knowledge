import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { DownloadProgressHandler } from '#lib/server/utils/download-progress';
import { downloadVerifiedFile } from '#lib/server/utils/verified-download';

export const LAYOUT_MODEL_DIR = resolve(process.cwd(), '.cache', 'layout');
export const LAYOUT_MODEL_PATH = join(LAYOUT_MODEL_DIR, 'model.onnx');
export const LAYOUT_MODEL_SIZE_BYTES = 171_220_471;

const MODEL_REVISION = '40bde044036bb181c130ddf6c51792187268748f';
const MODEL_URL = `https://huggingface.co/docling-project/docling-layout-heron-onnx/resolve/${MODEL_REVISION}/model.onnx`;
const MODEL_SHA256 = '59c81a3a2923042d85034ffc487f8f47e4854117e879aef89b2b9f728fb4922a';

let installation: Promise<void> | undefined;

export function isLayoutModelInstalled(): boolean {
	return existsSync(LAYOUT_MODEL_PATH);
}

export function installLayoutModel(onProgress?: DownloadProgressHandler): Promise<void> {
	if (isLayoutModelInstalled()) return Promise.resolve();

	installation ??= downloadVerifiedFile({
		url: MODEL_URL,
		destination: LAYOUT_MODEL_PATH,
		sha256: MODEL_SHA256,
		onProgress
	}).finally(() => {
		installation = undefined;
	});
	return installation;
}
