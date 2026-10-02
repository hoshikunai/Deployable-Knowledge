// Shared Transformers.js settings for every local model (embeddings and rerankers)

import { availableParallelism } from 'node:os';
import { resolve } from 'node:path';
import { env } from '@huggingface/transformers';

export const MODEL_CACHE_DIR = resolve(process.cwd(), '.cache', 'transformersjs');

export const INFERENCE_THREADS = Math.max(1, Math.min(8, Math.floor(availableParallelism() / 4)));

env.cacheDir = MODEL_CACHE_DIR;
env.localModelPath = MODEL_CACHE_DIR;
env.allowRemoteModels = true;
