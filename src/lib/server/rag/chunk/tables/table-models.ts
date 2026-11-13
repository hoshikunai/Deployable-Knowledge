import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { DownloadProgressHandler } from '#lib/server/utils/download-progress';
import { downloadVerifiedFile } from '#lib/server/utils/verified-download';

export const TABLE_MODEL_DIR = resolve(process.cwd(), '.cache', 'tables');

type ModelFile = { name: string; url: string; sha256: string; bytes: number };

const HERON_REVISION = '40bde044036bb181c130ddf6c51792187268748f';
const HERON_URL = `https://huggingface.co/docling-project/docling-layout-heron-onnx/resolve/${HERON_REVISION}/model.onnx`;
const TABLEFORMER_RELEASE =
	'https://github.com/docling-project/docling.rs/releases/download/models-v1';

const MODEL_FILES: ModelFile[] = [
	{
		name: 'layout_heron.onnx',
		url: HERON_URL,
		sha256: '59c81a3a2923042d85034ffc487f8f47e4854117e879aef89b2b9f728fb4922a',
		bytes: 171_220_471
	},
	{
		name: 'encoder.onnx',
		url: `${TABLEFORMER_RELEASE}/encoder.onnx`,
		sha256: '6042884931d4a5b0c651676e53d7b40d6658fbc3d0c197ed5bf37386ff8aa96b',
		bytes: 107_827_334
	},
	{
		name: 'decoder_kv.onnx',
		url: `${TABLEFORMER_RELEASE}/decoder_kv.onnx`,
		sha256: 'c87f0ff1c314179921c0332bbee6bb1207535643a132a4a9dc4905d3a3f573a5',
		bytes: 350_250
	},
	{
		name: 'decoder_kv.onnx.data',
		url: `${TABLEFORMER_RELEASE}/decoder_kv.onnx.data`,
		sha256: 'e71523a8b7603f983088b04c9d5cb4203be117e4a0e5950028ae46a9e98103eb',
		bytes: 115_580_928
	},
	{
		name: 'bbox.onnx',
		url: `${TABLEFORMER_RELEASE}/bbox.onnx`,
		sha256: 'a3cd88ae198bd8ae3338fbb89cf2f054474c964ae5001611727db7240eaf1125',
		bytes: 52_234
	},
	{
		name: 'bbox.onnx.data',
		url: `${TABLEFORMER_RELEASE}/bbox.onnx.data`,
		sha256: 'e936773730de30a2d10528f69e064fd9fd222259aa65078c421f3186b5b7e36a',
		bytes: 39_618_560
	}
];

export const TABLE_MODEL_PATHS = {
	layout: join(TABLE_MODEL_DIR, 'layout_heron.onnx'),
	encoder: join(TABLE_MODEL_DIR, 'encoder.onnx'),
	decoder: join(TABLE_MODEL_DIR, 'decoder_kv.onnx'),
	bbox: join(TABLE_MODEL_DIR, 'bbox.onnx')
};

export const TABLE_MODELS_SIZE_BYTES = MODEL_FILES.reduce((sum, file) => sum + file.bytes, 0);

let installation: Promise<void> | undefined;

export function areTableModelsInstalled(): boolean {
	return MODEL_FILES.every((file) => existsSync(join(TABLE_MODEL_DIR, file.name)));
}

async function downloadMissingFiles(onProgress?: DownloadProgressHandler): Promise<void> {
	let finishedBytes = 0;
	for (const file of MODEL_FILES) {
		const destination = join(TABLE_MODEL_DIR, file.name);
		if (!existsSync(destination)) {
			await downloadVerifiedFile({
				url: file.url,
				destination,
				sha256: file.sha256,
				onProgress: (loaded) => onProgress?.(finishedBytes + loaded, TABLE_MODELS_SIZE_BYTES)
			});
		}
		finishedBytes += file.bytes;
		onProgress?.(finishedBytes, TABLE_MODELS_SIZE_BYTES);
	}
}

export function installTableModels(onProgress?: DownloadProgressHandler): Promise<void> {
	if (areTableModelsInstalled()) return Promise.resolve();

	installation ??= downloadMissingFiles(onProgress).finally(() => {
		installation = undefined;
	});
	return installation;
}
