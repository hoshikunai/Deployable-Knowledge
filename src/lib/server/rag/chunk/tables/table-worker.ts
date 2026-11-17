import { createRequire } from 'node:module';
import { Worker } from 'node:worker_threads';
import { createLogger } from '#lib/server/logging';
import { INFERENCE_THREADS } from '#lib/server/utils/inference-threads';
import { TABLE_MODEL_PATHS } from './table-models';
import type { Box } from './table-types';

export const LAYOUT_INPUT_SIZE = 640;

const REQUEST_TIMEOUT_MS = 60_000;
const IDLE_TEARDOWN_MS = 60_000;

const log = createLogger('documents');

// onnxruntime's session.run blocks the event loop for the whole inference, so Heron and the
// three TableFormer graphs run here. The decode loop lives in the worker too: its KV cache
// grows every step and would otherwise be copied across threads per token.
//
// recognize() is a port of TableModel04_rs.predict and TFPredictor._prepare_image from
// docling-ibm-models (MIT): the crop is resized to 448x448 with OpenCV's bilinear sampling,
// normalized, and laid out (C, W, H) as Docling does; tags are decoded greedily with Docling's
// two structure corrections, and each cell's box comes from the decoder state Docling picks.
const WORKER_SOURCE = String.raw`
const { parentPort, workerData } = require("node:worker_threads");
const ort = require(workerData.ortPath);

const TABLE_LABEL = 8;
const MIN_TABLE_SCORE = 0.5;
const SIDE = 448;
const MEAN = [0.94247851, 0.94254675, 0.94292611];
const STD = [0.17910956, 0.17940403, 0.17931663];
const MAX_STEPS = 1024;
const MAX_ROW_TAGS = 256;
const LAYERS = 6;
const T = { start: 2, end: 3, ecel: 4, fcel: 5, lcel: 6, ucel: 7, xcel: 8, nl: 9, ched: 10, rhed: 11, srow: 12 };
const BOX_TAGS = new Set([T.fcel, T.ecel, T.ched, T.rhed, T.srow, T.nl, T.ucel]);

const options = {
	intraOpNumThreads: workerData.threads,
	interOpNumThreads: 1,
	graphOptimizationLevel: "all",
	enableCpuMemArena: false,
	enableMemPattern: false
};
let sessions;
const ready = Promise.all([
	ort.InferenceSession.create(workerData.paths.layout, options),
	ort.InferenceSession.create(workerData.paths.encoder, options),
	ort.InferenceSession.create(workerData.paths.decoder, options),
	ort.InferenceSession.create(workerData.paths.bbox, options)
]).then(
	([layout, encoder, decoder, bbox]) => {
		sessions = { layout, encoder, decoder, bbox };
		parentPort.postMessage({ type: "ready" });
	},
	(error) => parentPort.postMessage({ type: "fatal", message: String(error && error.message ? error.message : error) })
);

async function detect(message) {
	const rgba = message.pixels;
	const size = message.size;
	const plane = size * size;
	const planar = new Uint8Array(3 * plane);
	for (let index = 0; index < plane; index += 1) {
		planar[index] = rgba[index * 4];
		planar[plane + index] = rgba[index * 4 + 1];
		planar[2 * plane + index] = rgba[index * 4 + 2];
	}
	const sizes = BigInt64Array.from([BigInt(message.width), BigInt(message.height)]);
	const output = await sessions.layout.run({
		images: new ort.Tensor("uint8", planar, [1, 3, size, size]),
		orig_target_sizes: new ort.Tensor("int64", sizes, [1, 2])
	});
	const [labelName, boxName, scoreName] = sessions.layout.outputNames;
	const labels = output[labelName].data;
	const boxes = output[boxName].data;
	const scores = output[scoreName].data;
	const kept = [];
	for (let index = 0; index < scores.length; index += 1) {
		if (Number(labels[index]) === TABLE_LABEL && scores[index] >= MIN_TABLE_SCORE) kept.push(index);
	}
	const tableBoxes = new Float32Array(kept.length * 4);
	const tableScores = new Float32Array(kept.length);
	kept.forEach((source, target) => {
		tableBoxes.set(boxes.subarray(source * 4, source * 4 + 4), target * 4);
		tableScores[target] = scores[source];
	});
	return { result: { boxes: tableBoxes, scores: tableScores }, transfer: [tableBoxes.buffer, tableScores.buffer] };
}

function prepareImage(rgba, width, height) {
	const plane = SIDE * SIDE;
	const tensor = new Float32Array(3 * plane);
	const scaleX = width / SIDE;
	const scaleY = height / SIDE;
	for (let y = 0; y < SIDE; y += 1) {
		const sourceY = Math.max(0, (y + 0.5) * scaleY - 0.5);
		const y0 = Math.min(Math.floor(sourceY), height - 1);
		const y1 = Math.min(y0 + 1, height - 1);
		const wy = sourceY - y0;
		for (let x = 0; x < SIDE; x += 1) {
			const sourceX = Math.max(0, (x + 0.5) * scaleX - 0.5);
			const x0 = Math.min(Math.floor(sourceX), width - 1);
			const x1 = Math.min(x0 + 1, width - 1);
			const wx = sourceX - x0;
			for (let channel = 0; channel < 3; channel += 1) {
				const top = rgba[(y0 * width + x0) * 4 + channel] * (1 - wx) + rgba[(y0 * width + x1) * 4 + channel] * wx;
				const bottom = rgba[(y1 * width + x0) * 4 + channel] * (1 - wx) + rgba[(y1 * width + x1) * 4 + channel] * wx;
				const value = (top * (1 - wy) + bottom * wy) / 255;
				tensor[channel * plane + x * SIDE + y] = (value - MEAN[channel]) / STD[channel];
			}
		}
	}
	return new ort.Tensor("float32", tensor, [1, 3, SIDE, SIDE]);
}

function argmax(values) {
	let best = 0;
	for (let index = 1; index < values.length; index += 1) {
		if (values[index] > values[best]) best = index;
	}
	return best;
}

function mergeBoxes(first, last) {
	const width = last[0] + last[2] / 2 - (first[0] - first[2] / 2);
	const height = last[1] + last[3] / 2 - (first[1] - first[3] / 2);
	const left = first[0] - first[2] / 2;
	const top = Math.min(last[1] - last[3] / 2, first[1] - first[3] / 2);
	return [left + width / 2, top + height / 2, width, height];
}

async function recognize(message) {
	const encoded = await sessions.encoder.run({ image: prepareImage(message.pixels, message.width, message.height) });
	const cross = {};
	for (let layer = 0; layer < LAYERS; layer += 1) {
		cross["cross_kt_" + layer] = encoded["cross_kt_" + layer];
		cross["cross_v_" + layer] = encoded["cross_v_" + layer];
	}

	const empty = new ort.Tensor("float32", new Float32Array(0), [LAYERS, 1, 8, 0, 64]);
	let cacheK = empty;
	let cacheV = empty;
	let tag = T.start;
	const tags = [];
	const hidden = [];
	const merges = new Map();
	let skipNext = true;
	let previousUcel = false;
	let firstLcel = true;
	let spanStart = -1;
	let boxIndex = 0;
	let rowTags = 0;

	while (tags.length < MAX_STEPS) {
		const output = await sessions.decoder.run({
			...cross,
			tag: new ort.Tensor("int64", BigInt64Array.from([BigInt(tag)]), [1, 1]),
			cache_k: cacheK,
			cache_v: cacheV
		});
		cacheK = output.out_cache_k;
		cacheV = output.out_cache_v;
		const state = output.hidden.data;

		let next = argmax(output.logits.data);
		if (next === T.xcel) next = T.lcel;
		if (previousUcel && next === T.lcel) next = T.fcel;
		if (next === T.end) break;
		tags.push(next);

		rowTags = next === T.nl ? 0 : rowTags + 1;
		if (rowTags > MAX_ROW_TAGS) throw new Error("TableFormer produced a runaway row.");

		if (!skipNext && BOX_TAGS.has(next)) {
			hidden.push(state.slice());
			if (!firstLcel) merges.set(spanStart, boxIndex);
			boxIndex += 1;
		}
		if (next !== T.lcel) {
			firstLcel = true;
		} else if (firstLcel) {
			hidden.push(state.slice());
			firstLcel = false;
			spanStart = boxIndex;
			merges.set(spanStart, -1);
			boxIndex += 1;
		}
		skipNext = next === T.nl || next === T.ucel || next === T.xcel;
		previousUcel = next === T.ucel;
		tag = next;
	}

	if (!tags.includes(T.nl)) throw new Error("TableFormer found no table rows.");
	if (tags.length >= MAX_STEPS && tags[tags.length - 1] !== T.nl) {
		throw new Error("TableFormer hit the step limit mid-row.");
	}

	const count = hidden.length;
	let raw = new Float32Array(0);
	if (count > 0) {
		const states = new Float32Array(count * 512);
		hidden.forEach((state, index) => states.set(state, index * 512));
		const predicted = await sessions.bbox.run({
			enc_out: encoded.enc_out,
			tag_h: new ort.Tensor("float32", states, [count, 512])
		});
		raw = predicted.boxes.data;
	}

	const boxAt = (index) => Array.from(raw.subarray(index * 4, index * 4 + 4));
	const merged = [];
	const skipped = new Set();
	for (let index = 0; index < count; index += 1) {
		if (merges.has(index)) {
			const end = merges.get(index) === -1 ? count - 1 : merges.get(index);
			skipped.add(end);
			merged.push(mergeBoxes(boxAt(index), boxAt(end)));
		} else if (!skipped.has(index)) {
			merged.push(boxAt(index));
		}
	}

	const boxes = new Float32Array(merged.length * 4);
	merged.forEach(([cx, cy, w, h], index) => {
		boxes.set([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], index * 4);
	});
	const tagArray = Int32Array.from(tags);
	return { result: { tags: tagArray, boxes }, transfer: [tagArray.buffer, boxes.buffer] };
}

parentPort.on("message", async (message) => {
	await ready;
	if (!sessions) return;
	try {
		const { result, transfer } = message.type === "detect" ? await detect(message) : await recognize(message);
		parentPort.postMessage({ type: "result", id: message.id, ...result }, transfer);
	} catch (error) {
		parentPort.postMessage({ type: "error", id: message.id, message: String(error && error.message ? error.message : error) });
	}
});
`;

export type RgbaImage = { pixels: Uint8Array<ArrayBuffer>; width: number; height: number };

export type RecognizedTags = { tags: Int32Array; boxes: Float32Array };

type DetectResult = { boxes: Float32Array; scores: Float32Array };

type WorkerMessage =
	| { type: 'ready' }
	| { type: 'fatal'; message: string }
	| ({ type: 'result'; id: number } & (DetectResult | RecognizedTags))
	| { type: 'error'; id: number; message: string };

type PendingRequest = {
	resolve: (result: unknown) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
};

export class TableModelLoadError extends Error {}

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
		failPending(new TableModelLoadError(`Table models failed to load: ${message.message}`));
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
		workerData: { ortPath, paths: TABLE_MODEL_PATHS, threads: INFERENCE_THREADS }
	});
	created.on('message', handleMessage);
	created.on('error', (error: Error) => {
		log.error('Table worker crashed', {}, error);
		failPending(error);
		if (worker === created) worker = undefined;
	});
	created.on('exit', (code) => {
		failPending(new Error(`Table worker exited (code ${code}).`));
		if (worker === created) worker = undefined;
	});
	return created;
}

function scheduleIdleTeardown(): void {
	if (idleTimer) clearTimeout(idleTimer);
	idleTimer = setTimeout(stopWorker, IDLE_TEARDOWN_MS);
	idleTimer.unref?.();
}

function send<T>(message: Record<string, unknown>, pixels: Uint8Array<ArrayBuffer>): Promise<T> {
	worker ??= startWorker();
	const target = worker;
	const id = (nextRequestId += 1);

	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => {
			pending.delete(id);
			reject(new Error(`Table inference timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`));
			stopWorker();
		}, REQUEST_TIMEOUT_MS);
		pending.set(id, { resolve: (result) => resolve(result as T), reject, timer });
		target.postMessage({ ...message, id, pixels }, [pixels.buffer]);
	});
}

function enqueue<T>(run: () => Promise<T>): Promise<T> {
	const task = queue.then(async () => {
		try {
			return await run();
		} finally {
			scheduleIdleTeardown();
		}
	});
	queue = task.catch(() => undefined);
	return task;
}

/** Finds table regions on a 640x640 page render; boxes come back in page points. */
export function detectTables(image: RgbaImage, pageWidth: number, pageHeight: number) {
	return enqueue(async () => {
		const result = await send<DetectResult>(
			{
				type: 'detect',
				size: LAYOUT_INPUT_SIZE,
				width: Math.round(pageWidth),
				height: Math.round(pageHeight)
			},
			image.pixels
		);
		const tables: { box: Box; score: number }[] = [];
		result.scores.forEach((score, index) => {
			const [l, t, r, b] = result.boxes.subarray(index * 4, index * 4 + 4);
			tables.push({ box: { l, t, r, b }, score });
		});
		return tables;
	});
}

/** Reads the OTSL structure of one table crop; boxes are xyxy, normalized to the crop. */
export function recognizeTable(crop: RgbaImage): Promise<RecognizedTags> {
	return enqueue(() =>
		send<RecognizedTags>({ type: 'recognize', width: crop.width, height: crop.height }, crop.pixels)
	);
}
