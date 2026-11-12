import { createRequire } from 'node:module';
import { Worker } from 'node:worker_threads';
import { createLogger } from '#lib/server/logging';
import { INFERENCE_THREADS } from '#lib/server/utils/inference-threads';
import { LAYOUT_MODEL_PATH } from './heron-model';
import { LAYOUT_LABELS, type LayoutDetection, type LayoutInput } from './layout-types';

export const LAYOUT_INPUT_SIZE = 640;

const REQUEST_TIMEOUT_MS = 60_000;
const IDLE_TEARDOWN_MS = 60_000;

const log = createLogger('documents');

// Runs onnxruntime off the main thread because session.run blocks the event loop for the
// whole inference. Pixels arrive as 640x640 RGBA and are reordered to the planar uint8 RGB
// tensor the Heron export expects; scaling to [0, 1] happens inside the graph.
const WORKER_SOURCE = String.raw`
const { parentPort, workerData } = require("node:worker_threads");
const ort = require(workerData.ortPath);
const SIZE = workerData.size;
const MIN_SCORE = 0.3;

let session;
const ready = ort.InferenceSession.create(workerData.modelPath, {
	intraOpNumThreads: workerData.threads,
	interOpNumThreads: 1,
	graphOptimizationLevel: "all"
}).then(
	(created) => {
		session = created;
		parentPort.postMessage({ type: "ready" });
	},
	(error) => parentPort.postMessage({ type: "fatal", message: String(error && error.message ? error.message : error) })
);

parentPort.on("message", async (message) => {
	await ready;
	if (!session) return;
	try {
		const rgba = message.pixels;
		const plane = SIZE * SIZE;
		const planar = new Uint8Array(3 * plane);
		for (let index = 0; index < plane; index += 1) {
			planar[index] = rgba[index * 4];
			planar[plane + index] = rgba[index * 4 + 1];
			planar[2 * plane + index] = rgba[index * 4 + 2];
		}
		const sizes = BigInt64Array.from([BigInt(message.width), BigInt(message.height)]);
		const output = await session.run({
			images: new ort.Tensor("uint8", planar, [1, 3, SIZE, SIZE]),
			orig_target_sizes: new ort.Tensor("int64", sizes, [1, 2])
		});
		const [labelName, boxName, scoreName] = session.outputNames;
		const rawLabels = output[labelName].data;
		const rawBoxes = output[boxName].data;
		const rawScores = output[scoreName].data;
		const kept = [];
		for (let index = 0; index < rawScores.length; index += 1) {
			if (rawScores[index] >= MIN_SCORE) kept.push(index);
		}
		const labels = new Int32Array(kept.length);
		const scores = new Float32Array(kept.length);
		const boxes = new Float32Array(kept.length * 4);
		kept.forEach((source, target) => {
			labels[target] = Number(rawLabels[source]);
			scores[target] = rawScores[source];
			boxes.set(rawBoxes.subarray(source * 4, source * 4 + 4), target * 4);
		});
		parentPort.postMessage(
			{ type: "result", id: message.id, labels, scores, boxes },
			[labels.buffer, scores.buffer, boxes.buffer]
		);
	} catch (error) {
		parentPort.postMessage({ type: "error", id: message.id, message: String(error && error.message ? error.message : error) });
	}
});
`;

type WorkerResult = {
	type: 'result';
	id: number;
	labels: Int32Array;
	scores: Float32Array;
	boxes: Float32Array;
};

type WorkerMessage =
	| { type: 'ready' }
	| { type: 'fatal'; message: string }
	| WorkerResult
	| { type: 'error'; id: number; message: string };

type PendingRequest = {
	resolve: (result: WorkerResult) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
};

let worker: Worker | undefined;
let nextRequestId = 0;
let queue: Promise<unknown> = Promise.resolve();
let idleTimer: ReturnType<typeof setTimeout> | undefined;
const pending = new Map<number, PendingRequest>();

function failPending(error: Error): void {
	for (const request of pending.values()) {
		clearTimeout(request.timer);
		request.reject(error);
	}
	pending.clear();
}

function stopWorker(): void {
	const running = worker;
	worker = undefined;
	if (idleTimer) {
		clearTimeout(idleTimer);
		idleTimer = undefined;
	}
	void running?.terminate();
}

function handleMessage(message: WorkerMessage): void {
	if (message.type === 'ready') return;
	if (message.type === 'fatal') {
		failPending(new Error(`Layout model failed to load: ${message.message}`));
		stopWorker();
		return;
	}

	const request = pending.get(message.id);
	if (!request) return;
	pending.delete(message.id);
	clearTimeout(request.timer);
	if (message.type === 'error') request.reject(new Error(message.message));
	else request.resolve(message);
}

function startWorker(): Worker {
	const ortPath = createRequire(import.meta.url).resolve('onnxruntime-node');
	const created = new Worker(WORKER_SOURCE, {
		eval: true,
		execArgv: [],
		workerData: {
			ortPath,
			modelPath: LAYOUT_MODEL_PATH,
			threads: INFERENCE_THREADS,
			size: LAYOUT_INPUT_SIZE
		}
	});
	created.on('message', handleMessage);
	created.on('error', (error: Error) => {
		log.error('Layout worker crashed', {}, error);
		failPending(error);
		if (worker === created) worker = undefined;
	});
	created.on('exit', (code) => {
		failPending(new Error(`Layout worker exited (code ${code}).`));
		if (worker === created) worker = undefined;
	});
	return created;
}

function scheduleIdleTeardown(): void {
	if (idleTimer) clearTimeout(idleTimer);
	idleTimer = setTimeout(stopWorker, IDLE_TEARDOWN_MS);
	idleTimer.unref?.();
}

function runDetection(input: LayoutInput): Promise<WorkerResult> {
	worker ??= startWorker();
	const target = worker;
	const id = (nextRequestId += 1);

	return new Promise<WorkerResult>((resolve, reject) => {
		const timer = setTimeout(() => {
			pending.delete(id);
			reject(new Error(`Layout detection timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`));
			stopWorker();
		}, REQUEST_TIMEOUT_MS);
		pending.set(id, { resolve, reject, timer });
		target.postMessage(
			{
				type: 'detect',
				id,
				pixels: input.pixels,
				width: Math.round(input.width),
				height: Math.round(input.height)
			},
			[input.pixels.buffer]
		);
	});
}

function toDetections(result: WorkerResult): LayoutDetection[] {
	const detections: LayoutDetection[] = [];
	result.labels.forEach((labelIndex, index) => {
		const label = LAYOUT_LABELS[labelIndex];
		if (!label) return;
		const [l, t, r, b] = result.boxes.subarray(index * 4, index * 4 + 4);
		detections.push({ label, score: result.scores[index], box: { l, t, r, b } });
	});
	return detections;
}

export function detectLayout(input: LayoutInput): Promise<LayoutDetection[]> {
	const task = queue.then(async () => {
		try {
			return toDetections(await runDetection(input));
		} finally {
			scheduleIdleTeardown();
		}
	});
	queue = task.catch(() => undefined);
	return task;
}
