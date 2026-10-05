import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

export const RELEX_MODEL_DIR = resolve(process.cwd(), 'models', 'gliner-relex');

export const RELEX_MODEL_FILES = {
	encoder: 'encoder.onnx',
	adjacency: 'adjacency.onnx',
	pairs: 'pairs.onnx',
	tokenizer: 'tokenizer.json',
	tokenizerConfig: 'tokenizer_config.json',
	modelConfig: 'gliner_config.json'
} as const;

export type RelexModelFile = keyof typeof RELEX_MODEL_FILES;

export const RELEX_EXPORT_COMMAND =
	'benchmarks/gliner-parity/.venv/bin/python benchmarks/gliner-parity/relex_split.py export --out models/gliner-relex';

export class RelexModelMissingError extends Error {
	constructor(readonly missingFiles: string[]) {
		super(
			`The relation model is not installed: ${missingFiles.join(', ')} missing from ${RELEX_MODEL_DIR}. ` +
				`Create it with: ${RELEX_EXPORT_COMMAND}`
		);
		this.name = 'RelexModelMissingError';
	}
}

export function relexModelPath(file: RelexModelFile): string {
	return resolve(RELEX_MODEL_DIR, RELEX_MODEL_FILES[file]);
}

export function missingRelexModelFiles(): string[] {
	return Object.values(RELEX_MODEL_FILES).filter(
		(file) => !existsSync(resolve(RELEX_MODEL_DIR, file))
	);
}
