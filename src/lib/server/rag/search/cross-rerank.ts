// Ettin 32M cross-encoder relevance scorer.

import { resolve } from 'node:path';
import {
	AutoModel,
	AutoTokenizer,
	type Tensor
} from '@huggingface/transformers';
import { INFERENCE_THREADS } from '../embedding-model';
import {
	loadEttinHeadWeights,
	scoreEttinHiddenStates,
	type EttinHeadWeights
} from './ettin-head';

export const RERANKER_MODEL =
	'cross-encoder/ettin-reranker-32m-v1';

const MODEL_REVISION =
	'b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd';
const MODEL_FILE_NAME = 'model_quint8_avx2';
const MODEL_CACHE_DIR = resolve(
	process.cwd(),
	'.cache',
	'transformersjs'
);
const MAX_LENGTH = 512;
const RERANK_BATCH_SIZE = 8;

export type RerankCandidate = {
	chunkId: string;
	content: string;
};

export type RerankedCandidate = RerankCandidate & {
	relevance: number;
};

type Tokenizer = Awaited<
	ReturnType<typeof AutoTokenizer.from_pretrained>
>;
type Encoder = Awaited<
	ReturnType<typeof AutoModel.from_pretrained>
>;

type EttinRuntime = {
	tokenizer: Tokenizer;
	encoder: Encoder;
	head: EttinHeadWeights;
};

type EncoderOutput = {
	last_hidden_state?: Tensor;
};

let runtimePromise: Promise<EttinRuntime> | undefined;

function getRuntime(): Promise<EttinRuntime> {
	if (!runtimePromise) {
		console.log(`[Reranker] Loading ${RERANKER_MODEL}...`);
		const started = Date.now();

		runtimePromise = Promise.all([
			AutoTokenizer.from_pretrained(RERANKER_MODEL, {
				revision: MODEL_REVISION,
				cache_dir: MODEL_CACHE_DIR
			}),
			AutoModel.from_pretrained(RERANKER_MODEL, {
				revision: MODEL_REVISION,
				cache_dir: MODEL_CACHE_DIR,
				model_file_name: MODEL_FILE_NAME,
				dtype: 'fp32',
				device: 'cpu',
				session_options: {
					intraOpNumThreads: INFERENCE_THREADS,
					interOpNumThreads: 1
				}
			}),
			loadEttinHeadWeights()
		])
			.then(([tokenizer, encoder, head]) => {
				console.log(
					`[Reranker] ${RERANKER_MODEL} ready in ` +
						`${((Date.now() - started) / 1000).toFixed(1)}s.`
				);

				return { tokenizer, encoder, head };
			})
			.catch((error) => {
				runtimePromise = undefined;
				throw error;
			});
	}

	return runtimePromise;
}

async function scoreBatch(
	query: string,
	candidates: RerankCandidate[],
	runtime: EttinRuntime
): Promise<number[]> {
	const queries = new Array(candidates.length).fill(query);
	const passages = candidates.map(
		(candidate) => candidate.content
	);
	const inputs = await runtime.tokenizer(queries, {
		text_pair: passages,
		padding: true,
		truncation: true,
		max_length: MAX_LENGTH
	});
	let hiddenStates: Tensor | undefined;

	try {
		const output = (await runtime.encoder(
			inputs
		)) as EncoderOutput;
		hiddenStates = output.last_hidden_state;

		if (!hiddenStates) {
			throw new Error(
				'Ettin ONNX encoder did not return last_hidden_state'
			);
		}

		if (!(hiddenStates.data instanceof Float32Array)) {
			throw new Error(
				'Ettin ONNX encoder returned a non-float32 tensor'
			);
		}

		return scoreEttinHiddenStates(
			hiddenStates.data,
			Array.from(hiddenStates.dims),
			runtime.head
		);
	} finally {
		hiddenStates?.dispose();

		for (const tensor of Object.values(inputs)) {
			tensor.dispose();
		}
	}
}

export async function rerankCandidates(
	query: string,
	candidates: RerankCandidate[]
): Promise<RerankedCandidate[]> {
	const uniqueCandidates = [
		...new Map(
			candidates.map((candidate) => [
				candidate.chunkId,
				candidate
			])
		).values()
	];

	if (uniqueCandidates.length === 0) {
		return [];
	}

	const runtime = await getRuntime();
	const scores: number[] = [];

	for (
		let offset = 0;
		offset < uniqueCandidates.length;
		offset += RERANK_BATCH_SIZE
	) {
		const batch = uniqueCandidates.slice(
			offset,
			offset + RERANK_BATCH_SIZE
		);

		scores.push(
			...(await scoreBatch(query, batch, runtime))
		);
	}

	return uniqueCandidates
		.map((candidate, index) => ({
			candidate,
			score: scores[index]
		}))
		.sort((left, right) => right.score - left.score)
		.map(({ candidate, score }) => ({
			...candidate,
			relevance: 1 / (1 + Math.exp(-score))
		}));
}