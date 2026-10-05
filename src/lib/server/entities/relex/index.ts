import { DEFAULT_RELEX_THRESHOLDS, type RelexThresholds } from '$lib/constants';
import { decodeRelex, type RelexEntity, type RelexRelation } from './decode';
import { encodeRelexInputs, promptWords, splitWords } from './processor';
import { loadRelexRuntime, runRelex } from './runtime';

export {
	missingRelexModelFiles,
	RELEX_EXPORT_COMMAND,
	RelexModelMissingError
} from './model-files';
export { loadRelexRuntime } from './runtime';
export type { RelexEntity, RelexRelation } from './decode';

export interface RelexRequest {
	entityLabels: readonly string[];
	relationLabels: readonly string[];
	thresholds?: Partial<RelexThresholds>;
}

export interface RelexResult {
	entities: RelexEntity[];
	relations: RelexRelation[];
}

function cleanLabels(labels: readonly string[], kind: string): string[] {
	const cleaned = [...new Set(labels.map((label) => label.trim()))];
	if (cleaned.some((label) => !label)) throw new Error(`Relex ${kind} labels must not be empty.`);
	return cleaned;
}

export async function extractRelex(text: string, request: RelexRequest): Promise<RelexResult> {
	const entityLabels = cleanLabels(request.entityLabels, 'entity');
	const relationLabels = cleanLabels(request.relationLabels, 'relation');
	if (entityLabels.length === 0) throw new Error('Relex needs at least one entity label.');
	const thresholds = { ...DEFAULT_RELEX_THRESHOLDS, ...request.thresholds };
	const runtime = await loadRelexRuntime();
	const words = splitWords(text, runtime.config.maxWords);
	if (words.length === 0) return { entities: [], relations: [] };
	const inputs = encodeRelexInputs(
		runtime.tokenizer,
		runtime.special,
		runtime.config,
		words,
		promptWords(runtime.config, entityLabels, relationLabels)
	);
	const forward = await runRelex(runtime, inputs, thresholds);
	return decodeRelex(
		forward,
		words,
		text,
		{ entity: entityLabels, relation: relationLabels },
		thresholds
	);
}
