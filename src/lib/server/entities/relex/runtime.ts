import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-node';
import { PreTrainedTokenizer } from '@huggingface/transformers';
import { INFERENCE_THREADS } from '$lib/server/rag/inference';
import {
	missingRelexModelFiles,
	relexModelPath,
	RelexModelMissingError,
	type RelexModelFile
} from './model-files';
import type { RelexInputs, RelexModelConfig, RelexSpecialTokens } from './processor';

export interface RelexRuntime {
	tokenizer: PreTrainedTokenizer;
	special: RelexSpecialTokens;
	config: RelexModelConfig;
	encoder: ort.InferenceSession;
	adjacency: ort.InferenceSession;
	pairs: ort.InferenceSession;
}

export interface RelexForward {
	scores: Float32Array;
	words: number;
	maxWidth: number;
	entityClasses: number;
	entitySpans: Array<[number, number]>;
	pairs: Array<[number, number]>;
	relationLogits: Float32Array;
	relationClasses: number;
}

let runtimePromise: Promise<RelexRuntime> | undefined;

async function readJson(file: RelexModelFile): Promise<Record<string, unknown>> {
	const value: unknown = JSON.parse(await readFile(relexModelPath(file), 'utf8'));
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		throw new Error(`${relexModelPath(file)} is not a JSON object.`);
	}
	return value as Record<string, unknown>;
}

function requireNumber(source: Record<string, unknown>, key: string): number {
	const value = source[key];
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
		throw new Error(`gliner_config.json: '${key}' must be a positive integer.`);
	}
	return value;
}

function requireString(source: Record<string, unknown>, key: string): string {
	const value = source[key];
	if (typeof value !== 'string' || !value) {
		throw new Error(`Relex model config: '${key}' must be a non-empty string.`);
	}
	return value;
}

function readModelConfig(source: Record<string, unknown>): RelexModelConfig {
	return {
		maxWidth: requireNumber(source, 'max_width'),
		maxWords: requireNumber(source, 'max_len'),
		entityToken: requireString(source, 'ent_token'),
		relationToken: requireString(source, 'rel_token'),
		separatorToken: requireString(source, 'sep_token')
	};
}

function specialTokenId(tokenizer: PreTrainedTokenizer, token: string): number {
	const id = tokenizer.convert_tokens_to_ids(token);
	if (typeof id !== 'number' || id < 0) throw new Error(`Tokenizer has no id for '${token}'.`);
	return id;
}

function createSession(file: RelexModelFile): Promise<ort.InferenceSession> {
	return ort.InferenceSession.create(relexModelPath(file), {
		executionProviders: ['cpu'],
		intraOpNumThreads: INFERENCE_THREADS,
		interOpNumThreads: 1,
		graphOptimizationLevel: 'all'
	});
}

async function createRuntime(): Promise<RelexRuntime> {
	const missing = missingRelexModelFiles();
	if (missing.length > 0) throw new RelexModelMissingError(missing);
	const started = Date.now();
	console.log('[Relex] Loading the relation model...');
	const [tokenizerJson, tokenizerConfig, modelConfig] = await Promise.all([
		readJson('tokenizer'),
		readJson('tokenizerConfig'),
		readJson('modelConfig')
	]);
	const tokenizer = new PreTrainedTokenizer(tokenizerJson, tokenizerConfig);
	const [encoder, adjacency, pairs] = await Promise.all([
		createSession('encoder'),
		createSession('adjacency'),
		createSession('pairs')
	]);
	console.log(`[Relex] Ready in ${((Date.now() - started) / 1000).toFixed(1)}s.`);
	return {
		tokenizer,
		special: {
			cls: specialTokenId(tokenizer, requireString(tokenizerConfig, 'cls_token')),
			sep: specialTokenId(tokenizer, requireString(tokenizerConfig, 'sep_token'))
		},
		config: readModelConfig(modelConfig),
		encoder,
		adjacency,
		pairs
	};
}

export function loadRelexRuntime(): Promise<RelexRuntime> {
	runtimePromise ??= createRuntime().catch((error: unknown) => {
		runtimePromise = undefined;
		throw error;
	});
	return runtimePromise;
}

function sigmoid(value: number): number {
	return 1 / (1 + Math.exp(-value));
}

function float32Output(
	output: ort.Tensor | undefined,
	name: string
): ort.Tensor & { data: Float32Array } {
	if (!output || !(output.data instanceof Float32Array)) {
		throw new Error(`Relex graph output '${name}' is missing or not float32.`);
	}
	return output as ort.Tensor & { data: Float32Array };
}

function selectEntities(
	scores: Float32Array,
	spanRep: Float32Array,
	inputs: RelexInputs,
	entityClasses: number,
	dimension: number,
	threshold: number
): { representations: Float32Array; spans: Array<[number, number]> } {
	const kept: number[] = [];
	if (entityClasses > 0) {
		for (let span = 0; span < inputs.spanCount; span += 1) {
			if (!inputs.spanMask[span]) continue;
			let best = -Infinity;
			for (let label = 0; label < entityClasses; label += 1) {
				best = Math.max(best, scores[span * entityClasses + label]);
			}
			if (sigmoid(best) > threshold) kept.push(span);
		}
	}
	const representations = new Float32Array(kept.length * dimension);
	kept.forEach((span, entity) => {
		representations.set(
			spanRep.subarray(span * dimension, (span + 1) * dimension),
			entity * dimension
		);
	});
	const spans = kept.map((span): [number, number] => [
		Number(inputs.spanIdx[span * 2]),
		Number(inputs.spanIdx[span * 2 + 1])
	]);
	return { representations, spans };
}

function selectPairs(
	adjacency: Float32Array,
	entities: number,
	threshold: number
): Array<[number, number]> {
	const pairs: Array<[number, number]> = [];
	for (let head = 0; head < entities; head += 1) {
		for (let tail = 0; tail < entities; tail += 1) {
			if (head !== tail && adjacency[head * entities + tail] > threshold) pairs.push([head, tail]);
		}
	}
	return pairs;
}

function gatherRows(source: Float32Array, rows: number[], dimension: number): Float32Array {
	const gathered = new Float32Array(rows.length * dimension);
	rows.forEach((row, index) => {
		gathered.set(source.subarray(row * dimension, (row + 1) * dimension), index * dimension);
	});
	return gathered;
}

async function scorePairs(
	runtime: RelexRuntime,
	representations: Float32Array,
	pairs: Array<[number, number]>,
	relationPrompts: ort.Tensor,
	dimension: number
): Promise<Float32Array> {
	const dims = [1, pairs.length, dimension];
	const outputs = await runtime.pairs.run({
		head_rep: new ort.Tensor(
			'float32',
			gatherRows(
				representations,
				pairs.map(([head]) => head),
				dimension
			),
			dims
		),
		tail_rep: new ort.Tensor(
			'float32',
			gatherRows(
				representations,
				pairs.map(([, tail]) => tail),
				dimension
			),
			dims
		),
		rel_prompts: relationPrompts
	});
	return float32Output(outputs.rel_logits, 'rel_logits').data;
}

export async function runRelex(
	runtime: RelexRuntime,
	inputs: RelexInputs,
	thresholds: { entity: number; adjacency: number }
): Promise<RelexForward> {
	const encoded = await runtime.encoder.run({
		input_ids: new ort.Tensor('int64', inputs.inputIds, [1, inputs.tokenCount]),
		attention_mask: new ort.Tensor('int64', inputs.attentionMask, [1, inputs.tokenCount]),
		words_mask: new ort.Tensor('int64', inputs.wordsMask, [1, inputs.tokenCount]),
		text_lengths: new ort.Tensor('int64', inputs.textLengths, [1, 1]),
		span_idx: new ort.Tensor('int64', inputs.spanIdx, [1, inputs.spanCount, 2]),
		span_mask: new ort.Tensor('bool', inputs.spanMask, [1, inputs.spanCount])
	});
	const scores = float32Output(encoded.scores, 'scores');
	const spanRep = float32Output(encoded.span_rep, 'span_rep');
	const relationPrompts = float32Output(encoded.rel_prompts, 'rel_prompts');
	const entityClasses = Number(scores.dims[3]);
	const dimension = Number(spanRep.dims[3]);
	const relationClasses = Number(relationPrompts.dims[1]);
	const entities = selectEntities(
		scores.data,
		spanRep.data,
		inputs,
		entityClasses,
		dimension,
		thresholds.entity
	);
	const forward: RelexForward = {
		scores: scores.data,
		words: Number(scores.dims[1]),
		maxWidth: Number(scores.dims[2]),
		entityClasses,
		entitySpans: entities.spans,
		pairs: [],
		relationLogits: new Float32Array(0),
		relationClasses
	};
	if (entities.spans.length === 0 || relationClasses === 0) return forward;
	const adjacency = await runtime.adjacency.run({
		entity_rep: new ort.Tensor('float32', entities.representations, [
			1,
			entities.spans.length,
			dimension
		]),
		entity_mask: new ort.Tensor('int64', new BigInt64Array(entities.spans.length).fill(1n), [
			1,
			entities.spans.length
		])
	});
	forward.pairs = selectPairs(
		float32Output(adjacency.adjacency, 'adjacency').data,
		entities.spans.length,
		thresholds.adjacency
	);
	if (forward.pairs.length === 0) return forward;
	forward.relationLogits = await scorePairs(
		runtime,
		entities.representations,
		forward.pairs,
		relationPrompts,
		dimension
	);
	return forward;
}
