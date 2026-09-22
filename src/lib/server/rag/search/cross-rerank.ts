// Cross-encoder relevance scorer.

import { AutoModelForSequenceClassification, AutoTokenizer } from '@huggingface/transformers';
import { INFERENCE_THREADS } from '../embedding-model';

export type RerankCandidate = {
	chunkId: string;
	content: string;
};

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type ClassificationModel = Awaited<
	ReturnType<typeof AutoModelForSequenceClassification.from_pretrained>
>;

let tokenizer: Tokenizer | undefined;
let model: ClassificationModel | undefined;

async function initializeModel() {
	if (!tokenizer) {
		const modelId = 'Xenova/ms-marco-MiniLM-L-6-v2';
		tokenizer = await AutoTokenizer.from_pretrained(modelId);
	}
	if (!model) {
		const modelId = 'Xenova/ms-marco-MiniLM-L-6-v2';
		model = await AutoModelForSequenceClassification.from_pretrained(modelId, {
			session_options: { intraOpNumThreads: INFERENCE_THREADS, interOpNumThreads: 1 }
		});
	}

	return { tokenizer, model };
}

export type RerankedCandidate = RerankCandidate & { relevance: number };

const RERANK_BATCH_SIZE = 32;
const ETTIN_ENDPOINT = process.env.RAG_ETTIN_ENDPOINT ?? 'http://127.0.0.1:41792/rerank';
const ETTIN_TIMEOUT_MS = 120_000;
const ETTIN_MODELS = {
	'ettin-32m': 'cross-encoder/ettin-reranker-32m-v1',
	'ettin-68m': 'cross-encoder/ettin-reranker-68m-v1',
	'ettin-150m': 'cross-encoder/ettin-reranker-150m-v1',
	'ettin-400m': 'cross-encoder/ettin-reranker-400m-v1'
} as const;
export const RERANKER_PROVENANCE = {
	marco: { model: 'Xenova/ms-marco-MiniLM-L-6-v2', backend: 'transformers.js' },
	ettin: { models: ETTIN_MODELS, backend: 'sentence-transformers-torch-cpu' }
} as const;

type EttinResponse = { scores: unknown; model: unknown };

async function rerankWithEttin(
	query: string,
	candidates: RerankCandidate[],
	expectedModel: string
): Promise<RerankedCandidate[]> {
	const uniqueCandidates = [
		...new Map(candidates.map((candidate) => [candidate.chunkId, candidate])).values()
	];
	if (uniqueCandidates.length === 0) return [];

	const scores: number[] = [];
	for (let offset = 0; offset < uniqueCandidates.length; offset += RERANK_BATCH_SIZE) {
		const batch = uniqueCandidates.slice(offset, offset + RERANK_BATCH_SIZE);
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), ETTIN_TIMEOUT_MS);
		try {
			const response = await fetch(ETTIN_ENDPOINT, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ query, candidates: batch, maxLength: 512 }),
				signal: controller.signal
			});
			if (!response.ok) throw new Error(`HTTP ${response.status}`);
			const payload = (await response.json()) as EttinResponse;
			if (payload.model !== expectedModel)
				throw new Error(`expected model ${expectedModel}, received ${String(payload.model)}`);
			if (!Array.isArray(payload.scores) || payload.scores.length !== batch.length)
				throw new Error('score count does not match the candidate count');
			for (const [index, score] of payload.scores.entries()) {
				const value = Number(score);
				if (!Number.isFinite(value)) throw new Error(`non-finite score at index ${offset + index}`);
				scores.push(value);
			}
		} catch (error) {
			throw new Error(
				`Ettin reranker request failed (${ETTIN_ENDPOINT}): ${error instanceof Error ? error.message : String(error)}`
			);
		} finally {
			clearTimeout(timeout);
		}
	}
	return uniqueCandidates
		.map((candidate, index) => ({ candidate, score: scores[index] }))
		.sort((left, right) => right.score - left.score)
		.map(({ candidate, score }) => ({
			...candidate,
			relevance: 1 / (1 + Math.exp(-score))
		}));
}

export async function rerankWithMarco(
	query: string,
	candidates: RerankCandidate[]
): Promise<RerankedCandidate[]> {
	const uniqueCandidates = [
		...new Map(candidates.map((candidate) => [candidate.chunkId, candidate])).values()
	];

	if (uniqueCandidates.length === 0) return [];

	const { tokenizer, model } = await initializeModel();

	const scoredCandidates: { candidate: RerankCandidate; logit: number }[] = [];
	for (let offset = 0; offset < uniqueCandidates.length; offset += RERANK_BATCH_SIZE) {
		const batch = uniqueCandidates.slice(offset, offset + RERANK_BATCH_SIZE);
		const encodedInputs = await tokenizer(new Array(batch.length).fill(query), {
			text_pair: batch.map((candidate) => candidate.content),
			padding: true,
			truncation: true,
			max_length: 512
		});
		const { logits } = await model(encodedInputs);
		for (let index = 0; index < batch.length; index += 1) {
			scoredCandidates.push({ candidate: batch[index], logit: Number(logits.data[index]) });
		}
	}

	return scoredCandidates
		.sort((left, right) => right.logit - left.logit)
		.map(({ candidate, logit }) => ({
			...candidate,
			relevance: 1 / (1 + Math.exp(-logit))
		}));
}

export async function rerankCandidates(
	query: string,
	candidates: RerankCandidate[]
): Promise<RerankedCandidate[]> {
	const configuredModel = process.env.RAG_RERANK_MODEL?.trim();
	if (!configuredModel || configuredModel === 'marco') {
		return rerankWithMarco(query, candidates);
	}
	const expectedModel = ETTIN_MODELS[configuredModel as keyof typeof ETTIN_MODELS];
	if (!expectedModel) {
		throw new Error(`Unsupported RAG_RERANK_MODEL: ${configuredModel}`);
	}
	return rerankWithEttin(query, candidates, expectedModel);
}
