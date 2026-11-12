// src/lib/server/transcription/transcription-worker-entry.ts
import { parentPort } from "node:worker_threads";

// src/lib/server/transcription/diarization/pipeline.ts
import { InferenceSession } from "onnxruntime-node";

// src/lib/server/transcription/diarization/assets.ts
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
var MODEL_DIR = resolve(process.cwd(), "models", "speaker-diarization");
var MODEL_FILES = {
  segmentation: {
    path: "segmentation-3.0.onnx",
    url: "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx",
    sha256: "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25"
  },
  embedding: {
    path: "ecapa-tdnn.onnx",
    url: "https://huggingface.co/astrolabos/ecapa-tdnn/resolve/a97d2ae135a50c79d9bc6f1b8353700590369645/ecapa_tdnn.onnx",
    sha256: "9acb086d5d9efca07266930ce03772f13cad5fd5a415bcf014ee4d82bed26311"
  }
};
function assertDigest(bytes, model, origin) {
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== model.sha256) {
    throw new Error(`Diarization model ${origin} has sha256 ${digest}; expected ${model.sha256}.`);
  }
}
async function downloadModel(model, path) {
  const response = await fetch(model.url);
  if (!response.ok) {
    throw new Error(`Could not download diarization model ${model.url}: HTTP ${response.status}.`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  assertDigest(bytes, model, model.url);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(`${path}.download`, bytes);
  await rename(`${path}.download`, path);
  return bytes;
}
async function loadModel(model) {
  const path = resolve(MODEL_DIR, model.path);
  let file;
  try {
    file = await readFile(path);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    return downloadModel(model, path);
  }
  const bytes = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);
  assertDigest(bytes, model, path);
  return bytes;
}
async function loadDiarizationModels() {
  const [segmentation, embedding] = await Promise.all([
    loadModel(MODEL_FILES.segmentation),
    loadModel(MODEL_FILES.embedding)
  ]);
  return { segmentation, embedding };
}

// src/lib/server/transcription/diarization/clustering.ts
import { totalmem } from "node:os";

// src/lib/server/transcription/diarization/ahc.ts
function condensedIndex(n, i, j) {
  return n * i - i * (i + 1) / 2 + (j - i - 1);
}
function centroidLinkage(points, count, dimension) {
  const distances = new Float32Array(count * (count - 1) / 2);
  let pair = 0;
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      let dot = 0;
      for (let d = 0; d < dimension; d++)
        dot += points[i * dimension + d] * points[j * dimension + d];
      distances[pair++] = Math.sqrt(Math.max(0, 2 - 2 * dot));
    }
  }
  const size = new Float64Array(count).fill(1);
  const clusterId = Int32Array.from({ length: count }, (_, index) => index);
  const neighbor = new Int32Array(count).fill(-1);
  const minDistance = new Float64Array(count).fill(Infinity);
  const findNearest = (x) => {
    let best = -1;
    let bestDistance = Infinity;
    for (let z = x + 1; z < count; z++) {
      if (size[z] === 0) continue;
      const distance = distances[condensedIndex(count, x, z)];
      if (distance < bestDistance) {
        best = z;
        bestDistance = distance;
      }
    }
    neighbor[x] = best;
    minDistance[x] = bestDistance;
  };
  const closestCluster = () => {
    let best = -1;
    for (let x = 0; x < count - 1; x++) {
      if (size[x] > 0 && (best < 0 || minDistance[x] < minDistance[best])) best = x;
    }
    return best;
  };
  for (let x = 0; x < count - 1; x++) findNearest(x);
  const merges = [];
  for (let step = 0; step < count - 1; step++) {
    let x = closestCluster();
    let y = neighbor[x];
    while (minDistance[x] !== distances[condensedIndex(count, x, y)]) {
      findNearest(x);
      x = closestCluster();
      y = neighbor[x];
    }
    const distance = minDistance[x];
    const sizeX = size[x];
    const sizeY = size[y];
    const merged = sizeX + sizeY;
    merges.push({
      left: Math.min(clusterId[x], clusterId[y]),
      right: Math.max(clusterId[x], clusterId[y]),
      distance
    });
    size[x] = 0;
    size[y] = merged;
    clusterId[y] = count + step;
    for (let z = 0; z < count; z++) {
      if (size[z] === 0 || z === y) continue;
      const dxz = z < x ? distances[condensedIndex(count, z, x)] : distances[condensedIndex(count, x, z)];
      const yzIndex = z < y ? condensedIndex(count, z, y) : condensedIndex(count, y, z);
      const dyz = distances[yzIndex];
      const squared = (sizeX * dxz * dxz + sizeY * dyz * dyz - sizeX * sizeY * distance * distance / merged) / merged;
      distances[yzIndex] = Math.sqrt(Math.max(0, squared));
    }
    for (let z = 0; z < x; z++) {
      if (size[z] > 0 && neighbor[z] === x) neighbor[z] = y;
    }
    for (let z = 0; z < y; z++) {
      if (size[z] === 0) continue;
      const candidate = distances[condensedIndex(count, z, y)];
      if (candidate < minDistance[z]) {
        neighbor[z] = y;
        minDistance[z] = candidate;
      }
    }
    if (y < count - 1) findNearest(y);
  }
  return merges;
}
function cutDendrogram(merges, count, threshold) {
  const maxDistance = new Float64Array(merges.length);
  merges.forEach((merge, index) => {
    let value = merge.distance;
    if (merge.left >= count) value = Math.max(value, maxDistance[merge.left - count]);
    if (merge.right >= count) value = Math.max(value, maxDistance[merge.right - count]);
    maxDistance[index] = value;
  });
  const labels = new Int32Array(count);
  const stack = [[2 * count - 2, 0]];
  let clusters = 0;
  while (stack.length > 0) {
    const [node, inherited] = stack.pop();
    if (node < count) {
      labels[node] = inherited || ++clusters;
      continue;
    }
    const { left, right } = merges[node - count];
    const label = inherited || (maxDistance[node - count] <= threshold ? ++clusters : 0);
    for (const child of [right, left]) if (child < count) stack.push([child, label]);
    for (const child of [right, left]) if (child >= count) stack.push([child, label]);
  }
  return labels;
}
function agglomerativeClusters(points, count, dimension, threshold) {
  const flat = cutDendrogram(centroidLinkage(points, count, dimension), count, threshold);
  const unique = [...new Set(flat)].sort((left, right) => left - right);
  const relabel = new Map(unique.map((label, index) => [label, index]));
  return flat.map((label) => relabel.get(label));
}

// src/lib/server/transcription/diarization/embeddings.ts
import { Tensor as Tensor2 } from "onnxruntime-node";

// src/lib/server/transcription/diarization/fbank.ts
var MEL_BINS = 80;
var SAMPLE_RATE = 16e3;
var FFT_SIZE = 400;
var FBANK_HOP = 160;
var PADDING = FFT_SIZE / 2;
var SPECTRUM_BINS = FFT_SIZE / 2 + 1;
var OUTER = 16;
var INNER = 25;
var AMPLITUDE_FLOOR = 1e-10;
var DYNAMIC_RANGE_DB = 80;
var window = Float64Array.from(
  { length: FFT_SIZE },
  (_, n) => 0.54 - 0.46 * Math.cos(2 * Math.PI * n / FFT_SIZE)
);
function rootsOfUnity(size, rows, columns) {
  const cos = new Float64Array(rows * columns);
  const sin = new Float64Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const angle = -2 * Math.PI * (row * column % size) / size;
      cos[row * columns + column] = Math.cos(angle);
      sin[row * columns + column] = Math.sin(angle);
    }
  }
  return { cos, sin };
}
var outerRoots = rootsOfUnity(OUTER, OUTER, OUTER);
var twiddles = rootsOfUnity(FFT_SIZE, INNER, OUTER);
var innerRoots = rootsOfUnity(INNER, INNER, INNER);
function toMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function toHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}
var melFilters = (() => {
  const melLow = toMel(0);
  const melHigh = toMel(SAMPLE_RATE / 2);
  const hz = Array.from(
    { length: MEL_BINS + 2 },
    (_, index) => toHz(melLow + (melHigh - melLow) * index / (MEL_BINS + 1))
  );
  return Array.from({ length: MEL_BINS }, (_, bin) => {
    const center = hz[bin + 1];
    const band = hz[bin + 1] - hz[bin];
    const weights = [];
    let first = -1;
    for (let spectrumBin = 0; spectrumBin < SPECTRUM_BINS; spectrumBin++) {
      const frequency = spectrumBin * (SAMPLE_RATE / 2) / (SPECTRUM_BINS - 1);
      const slope = (frequency - center) / band;
      const weight = Math.max(0, Math.min(1 + slope, 1 - slope));
      if (weight === 0) continue;
      if (first < 0) first = spectrumBin;
      weights[spectrumBin - first] = weight;
    }
    return { first, weights: Float64Array.from(weights, (weight) => weight ?? 0) };
  });
})();
var real = new Float64Array(INNER * OUTER);
var imaginary = new Float64Array(INNER * OUTER);
var frame = new Float64Array(FFT_SIZE);
var power = new Float64Array(SPECTRUM_BINS);
var means = new Float64Array(MEL_BINS);
function powerSpectrum() {
  for (let n2 = 0; n2 < INNER; n2++) {
    for (let k1 = 0; k1 < OUTER; k1++) {
      let sumReal = 0;
      let sumImaginary = 0;
      for (let n1 = 0; n1 < OUTER; n1++) {
        const sample = frame[INNER * n1 + n2];
        sumReal += sample * outerRoots.cos[n1 * OUTER + k1];
        sumImaginary += sample * outerRoots.sin[n1 * OUTER + k1];
      }
      const twiddle = n2 * OUTER + k1;
      real[twiddle] = sumReal * twiddles.cos[twiddle] - sumImaginary * twiddles.sin[twiddle];
      imaginary[twiddle] = sumReal * twiddles.sin[twiddle] + sumImaginary * twiddles.cos[twiddle];
    }
  }
  for (let k1 = 0; k1 < OUTER; k1++) {
    for (let k2 = 0; k1 + OUTER * k2 < SPECTRUM_BINS; k2++) {
      let sumReal = 0;
      let sumImaginary = 0;
      for (let n2 = 0; n2 < INNER; n2++) {
        const value = n2 * OUTER + k1;
        const root = n2 * INNER + k2;
        sumReal += real[value] * innerRoots.cos[root] - imaginary[value] * innerRoots.sin[root];
        sumImaginary += real[value] * innerRoots.sin[root] + imaginary[value] * innerRoots.cos[root];
      }
      power[k1 + OUTER * k2] = sumReal * sumReal + sumImaginary * sumImaginary;
    }
  }
}
function countFbankFrames(sampleCount) {
  return 1 + Math.floor(sampleCount / FBANK_HOP);
}
function computeLogMelFrames(samples, firstFrame, frameCount, target) {
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
    const start = (firstFrame + frameIndex) * FBANK_HOP - PADDING;
    for (let n = 0; n < FFT_SIZE; n++) {
      const index = start + n;
      const sample = index >= 0 && index < samples.length ? samples[index] : 0;
      frame[n] = sample * window[n];
    }
    powerSpectrum();
    for (let bin = 0; bin < MEL_BINS; bin++) {
      const { first, weights } = melFilters[bin];
      let energy = 0;
      for (let offset = 0; offset < weights.length; offset++)
        energy += power[first + offset] * weights[offset];
      target[frameIndex * MEL_BINS + bin] = 10 * Math.log10(Math.max(energy, AMPLITUDE_FLOOR));
    }
  }
}
function normalizeFbank(logMel, frameCount, output) {
  const valueCount = frameCount * MEL_BINS;
  if (output.length < valueCount) {
    throw new Error(`Fbank output holds ${output.length} values; ${valueCount} are needed.`);
  }
  let maximum = -Infinity;
  for (let index = 0; index < valueCount; index++) {
    if (logMel[index] > maximum) maximum = logMel[index];
  }
  const floor = maximum - DYNAMIC_RANGE_DB;
  means.fill(0);
  for (let index = 0; index < valueCount; index++) {
    means[index % MEL_BINS] += Math.max(logMel[index], floor);
  }
  for (let bin = 0; bin < MEL_BINS; bin++) means[bin] /= frameCount;
  for (let index = 0; index < valueCount; index++) {
    output[index] = Math.max(logMel[index], floor) - means[index % MEL_BINS];
  }
  return output.subarray(0, valueCount);
}

// src/lib/server/transcription/diarization/segmentation.ts
import { Tensor } from "onnxruntime-node";
var SAMPLE_RATE2 = 16e3;
var CHUNK_SAMPLES = 10 * SAMPLE_RATE2;
var STEP_SAMPLES = SAMPLE_RATE2;
var FRAMES_PER_CHUNK = 589;
var LOCAL_SPEAKERS = 3;
var CHUNK_WINDOW = { start: 0, duration: 10, step: 1 };
var FRAME_WINDOW = { start: 0, duration: 0.0619375, step: 0.016875 };
var POWERSET_CLASSES = 7;
var SEGMENTATION_BATCH = 16;
var POWERSET_MAPPING = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 0],
  [1, 0, 1],
  [0, 1, 1]
];
function countChunks(sampleCount) {
  const fullChunks = sampleCount >= CHUNK_SAMPLES ? Math.floor((sampleCount - CHUNK_SAMPLES) / STEP_SAMPLES) + 1 : 0;
  const hasLastChunk = sampleCount < CHUNK_SAMPLES || (sampleCount - CHUNK_SAMPLES) % STEP_SAMPLES > 0;
  return fullChunks + (hasLastChunk ? 1 : 0);
}
function copyChunk(samples, index, target) {
  const start = index * STEP_SAMPLES;
  const available = samples.subarray(start, Math.min(samples.length, start + CHUNK_SAMPLES));
  target.set(available);
  target.fill(0, available.length);
}
function decodeBatch(scores, frameCount, activity, firstFrame) {
  for (let frame2 = 0; frame2 < frameCount; frame2++) {
    let best = 0;
    for (let powersetClass = 1; powersetClass < POWERSET_CLASSES; powersetClass++) {
      if (scores[frame2 * POWERSET_CLASSES + powersetClass] > scores[frame2 * POWERSET_CLASSES + best]) {
        best = powersetClass;
      }
    }
    activity.set(POWERSET_MAPPING[best], (firstFrame + frame2) * LOCAL_SPEAKERS);
  }
}
async function runSegmentation(session, samples) {
  const chunkCount = countChunks(samples.length);
  const activity = new Uint8Array(chunkCount * FRAMES_PER_CHUNK * LOCAL_SPEAKERS);
  const input = new Float32Array(Math.min(SEGMENTATION_BATCH, chunkCount) * CHUNK_SAMPLES);
  for (let first = 0; first < chunkCount; first += SEGMENTATION_BATCH) {
    const batch = Math.min(SEGMENTATION_BATCH, chunkCount - first);
    const batchInput = input.subarray(0, batch * CHUNK_SAMPLES);
    for (let offset = 0; offset < batch; offset++) {
      copyChunk(
        samples,
        first + offset,
        batchInput.subarray(offset * CHUNK_SAMPLES, (offset + 1) * CHUNK_SAMPLES)
      );
    }
    const output = await session.run({
      input_values: new Tensor("float32", batchInput, [batch, 1, CHUNK_SAMPLES])
    });
    const batchScores = output.logits;
    if (batchScores.dims.join("x") !== `${batch}x${FRAMES_PER_CHUNK}x${POWERSET_CLASSES}`) {
      throw new Error(`Unexpected segmentation output shape ${batchScores.dims.join("x")}.`);
    }
    decodeBatch(
      batchScores.data,
      batch * FRAMES_PER_CHUNK,
      activity,
      first * FRAMES_PER_CHUNK
    );
  }
  return { chunkCount, activity };
}

// src/lib/server/transcription/diarization/embeddings.ts
var EMBEDDING_DIMENSION = 192;
var EMBEDDING_INPUT = "feats";
var EMBEDDING_OUTPUT = "emb";
var MIN_EMBEDDING_SAMPLES = 640;
var MIN_EMBEDDING_FRAMES = countFbankFrames(MIN_EMBEDDING_SAMPLES);
var MIN_CLEAN_FRAMES = Math.ceil(FRAMES_PER_CHUNK * MIN_EMBEDDING_SAMPLES / CHUNK_SAMPLES);
var PROGRESS_INTERVAL = 16;
var FBANK_FRAMES_PER_CHUNK = countFbankFrames(CHUNK_SAMPLES);
var FBANK_FRAMES_PER_STEP = STEP_SAMPLES / FBANK_HOP;
var FBANK_FRAMES_KEPT = FBANK_FRAMES_PER_CHUNK - FBANK_FRAMES_PER_STEP;
var maskScale = Math.fround(FRAMES_PER_CHUNK / CHUNK_SAMPLES);
var frameOfSample = Int32Array.from(
  { length: CHUNK_SAMPLES },
  (_, sample) => Math.min(FRAMES_PER_CHUNK - 1, Math.floor(Math.fround(sample * maskScale)))
);
var samplesPerMaskFrame = new Int32Array(FRAMES_PER_CHUNK);
for (const frame2 of frameOfSample) samplesPerMaskFrame[frame2]++;
var maskFrameOfFbankFrame = Int32Array.from(
  { length: FBANK_FRAMES_PER_CHUNK },
  (_, frame2) => frameOfSample[Math.min(frame2 * FBANK_HOP, CHUNK_SAMPLES - 1)]
);
function speakerMask(activity, chunkOffset, speaker) {
  const full = new Uint8Array(FRAMES_PER_CHUNK);
  const clean = new Uint8Array(FRAMES_PER_CHUNK);
  let cleanFrames = 0;
  for (let frame2 = 0; frame2 < FRAMES_PER_CHUNK; frame2++) {
    const base = chunkOffset + frame2 * LOCAL_SPEAKERS;
    let active = 0;
    for (let local = 0; local < LOCAL_SPEAKERS; local++) active += activity[base + local];
    full[frame2] = activity[base + speaker];
    if (active < 2) {
      clean[frame2] = full[frame2];
      cleanFrames += clean[frame2];
    }
  }
  return cleanFrames > MIN_CLEAN_FRAMES ? clean : full;
}
function advanceLogMelWindow(samples, chunk, logMel) {
  if (chunk === 0) {
    computeLogMelFrames(samples, 0, FBANK_FRAMES_PER_CHUNK, logMel);
    return;
  }
  logMel.copyWithin(0, FBANK_FRAMES_PER_STEP * MEL_BINS);
  computeLogMelFrames(
    samples,
    chunk * FBANK_FRAMES_PER_STEP + FBANK_FRAMES_KEPT,
    FBANK_FRAMES_PER_STEP,
    logMel.subarray(FBANK_FRAMES_KEPT * MEL_BINS)
  );
}
function selectMaskedFrames(logMel, mask, selected) {
  let frameCount = 0;
  for (let frame2 = 0; frame2 < FBANK_FRAMES_PER_CHUNK; frame2++) {
    if (!mask[maskFrameOfFbankFrame[frame2]]) continue;
    selected.set(logMel.subarray(frame2 * MEL_BINS, (frame2 + 1) * MEL_BINS), frameCount * MEL_BINS);
    frameCount++;
  }
  return frameCount;
}
async function embedFeatures(session, features, frameCount, target) {
  const output = await session.run({
    [EMBEDDING_INPUT]: new Tensor2("float32", features, [1, frameCount, MEL_BINS])
  });
  const embedding = output[EMBEDDING_OUTPUT];
  if (embedding.data.length !== EMBEDDING_DIMENSION) {
    throw new Error(`Unexpected speaker embedding shape ${embedding.dims.join("x")}.`);
  }
  target.set(embedding.data);
}
async function extractEmbeddings(session, samples, activity, chunkCount, onProgress) {
  const embeddings = new Float64Array(chunkCount * LOCAL_SPEAKERS * EMBEDDING_DIMENSION).fill(NaN);
  const logMel = new Float64Array(FBANK_FRAMES_PER_CHUNK * MEL_BINS);
  const selected = new Float64Array(FBANK_FRAMES_PER_CHUNK * MEL_BINS);
  const features = new Float32Array(FBANK_FRAMES_PER_CHUNK * MEL_BINS);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    advanceLogMelWindow(samples, chunk, logMel);
    for (let speaker = 0; speaker < LOCAL_SPEAKERS; speaker++) {
      const mask = speakerMask(activity, chunk * FRAMES_PER_CHUNK * LOCAL_SPEAKERS, speaker);
      let sampleCount = 0;
      for (let frame2 = 0; frame2 < FRAMES_PER_CHUNK; frame2++) {
        if (mask[frame2]) sampleCount += samplesPerMaskFrame[frame2];
      }
      if (sampleCount < MIN_EMBEDDING_SAMPLES) continue;
      const frameCount = selectMaskedFrames(logMel, mask, selected);
      if (frameCount < MIN_EMBEDDING_FRAMES) continue;
      const start = (chunk * LOCAL_SPEAKERS + speaker) * EMBEDDING_DIMENSION;
      await embedFeatures(
        session,
        normalizeFbank(selected, frameCount, features),
        frameCount,
        embeddings.subarray(start, start + EMBEDDING_DIMENSION)
      );
    }
    if ((chunk + 1) % PROGRESS_INTERVAL === 0 || chunk === chunkCount - 1) {
      onProgress?.((chunk + 1) / chunkCount);
    }
  }
  return embeddings;
}

// src/lib/server/transcription/diarization/rounding.ts
function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
function closestFrame(time, window2) {
  return roundHalfEven((time - window2.start - 0.5 * window2.duration) / window2.step);
}

// src/lib/server/transcription/diarization/clustering.ts
var UNASSIGNED = -2;
var MIN_ACTIVE_RATIO = 0.2;
var THRESHOLD = 0.75;
var MIN_CLUSTER_SIZE = 12;
var ORPHAN_DISTANCE = 0.8;
var MIN_SPEAKER_SAMPLES = 6;
var MAX_DISTANCE_TABLE_SHARE_OF_RAM = 0.25;
function assertDistanceTableFits(sampleCount) {
  const bytes = sampleCount * (sampleCount - 1) / 2 * Float32Array.BYTES_PER_ELEMENT;
  const limit = totalmem() * MAX_DISTANCE_TABLE_SHARE_OF_RAM;
  if (bytes <= limit) return;
  const gb = (value) => (value / 1024 ** 3).toFixed(1);
  throw new Error(
    `Too many voice samples (${sampleCount}) to identify speakers: clustering would need ${gb(bytes)} GB, more than the ${gb(limit)} GB allowed on this machine.`
  );
}
function embeddingAt(embeddings, index) {
  return embeddings.subarray(index * EMBEDDING_DIMENSION, (index + 1) * EMBEDDING_DIMENSION);
}
function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let d = 0; d < EMBEDDING_DIMENSION; d++) {
    dot += left[d] * right[d];
    leftNorm += left[d] * left[d];
    rightNorm += right[d] * right[d];
  }
  return dot / Math.sqrt(leftNorm * rightNorm);
}
function cosineDistance(left, right) {
  return 1 - cosineSimilarity(left, right);
}
function groupOrphans(orphans, centroids) {
  const parent = new Map(orphans.map((label) => [label, label]));
  const root = (label) => {
    while (parent.get(label) !== label) label = parent.get(label);
    return label;
  };
  orphans.forEach((left, index) => {
    for (const right of orphans.slice(index + 1)) {
      if (cosineDistance(centroids.get(left), centroids.get(right)) > ORPHAN_DISTANCE) continue;
      const leftRoot = root(left);
      const rightRoot = root(right);
      parent.set(Math.max(leftRoot, rightRoot), Math.min(leftRoot, rightRoot));
    }
  });
  return new Map(orphans.map((label) => [label, root(label)]));
}
function mostSimilar(embedding, centroids) {
  let best = 0;
  let bestScore = cosineSimilarity(embedding, centroids[0]);
  for (let cluster = 1; cluster < centroids.length; cluster++) {
    const score = cosineSimilarity(embedding, centroids[cluster]);
    if (score > bestScore) {
      best = cluster;
      bestScore = score;
    }
  }
  return best;
}
function clusterCentroid(embeddings, labels, label) {
  const centroid = new Float64Array(EMBEDDING_DIMENSION);
  let members = 0;
  labels.forEach((rowLabel, row) => {
    if (rowLabel !== label) return;
    members++;
    const embedding = embeddingAt(embeddings, row);
    for (let d = 0; d < EMBEDDING_DIMENSION; d++) centroid[d] += embedding[d];
  });
  return centroid.map((value) => value / members);
}
function activityTotals(activity, chunkCount) {
  const active = new Int32Array(chunkCount * LOCAL_SPEAKERS);
  const clean = new Int32Array(chunkCount * LOCAL_SPEAKERS);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    for (let frame2 = 0; frame2 < FRAMES_PER_CHUNK; frame2++) {
      const base = (chunk * FRAMES_PER_CHUNK + frame2) * LOCAL_SPEAKERS;
      let speaking = 0;
      for (let speaker = 0; speaker < LOCAL_SPEAKERS; speaker++)
        speaking += activity[base + speaker];
      for (let speaker = 0; speaker < LOCAL_SPEAKERS; speaker++) {
        if (!activity[base + speaker]) continue;
        active[chunk * LOCAL_SPEAKERS + speaker]++;
        if (speaking === 1) clean[chunk * LOCAL_SPEAKERS + speaker]++;
      }
    }
  }
  return { active, clean };
}
function clusterTraining(trainEmbeddings, count) {
  const normalized = new Float64Array(trainEmbeddings.length);
  for (let row = 0; row < count; row++) {
    const embedding = embeddingAt(trainEmbeddings, row);
    const norm = Math.hypot(...embedding);
    for (let d = 0; d < EMBEDDING_DIMENSION; d++)
      normalized[row * EMBEDDING_DIMENSION + d] = embedding[d] / norm;
  }
  assertDistanceTableFits(count);
  const labels = agglomerativeClusters(normalized, count, EMBEDDING_DIMENSION, THRESHOLD);
  const minClusterSize = Math.min(MIN_CLUSTER_SIZE, Math.max(1, roundHalfEven(0.1 * count)));
  const sizes = /* @__PURE__ */ new Map();
  for (const label of labels) sizes.set(label, (sizes.get(label) ?? 0) + 1);
  const sortedLabels = [...sizes.keys()].sort((left, right) => left - right);
  const large = sortedLabels.filter((label) => sizes.get(label) >= minClusterSize);
  if (large.length === 0) return new Int32Array(count);
  const target = new Map(large.map((label, index) => [label, index]));
  const largeCentroids = large.map((label) => clusterCentroid(normalized, labels, label));
  const small = sortedLabels.filter((label) => !target.has(label));
  const centroids = new Map(
    small.map((label) => [label, clusterCentroid(normalized, labels, label)])
  );
  const nearestLarge = new Map(
    small.map((label) => [label, mostSimilar(centroids.get(label), largeCentroids)])
  );
  const orphans = small.filter(
    (label) => cosineDistance(centroids.get(label), largeCentroids[nearestLarge.get(label)]) > ORPHAN_DISTANCE
  );
  const groups = groupOrphans(orphans, centroids);
  const groupSizes = /* @__PURE__ */ new Map();
  for (const label of orphans) {
    const group = groups.get(label);
    groupSizes.set(group, (groupSizes.get(group) ?? 0) + sizes.get(label));
  }
  const newSpeakers = /* @__PURE__ */ new Map();
  for (const label of small) {
    const group = groups.get(label);
    if (group === void 0 || groupSizes.get(group) < MIN_SPEAKER_SAMPLES) {
      target.set(label, nearestLarge.get(label));
      continue;
    }
    if (!newSpeakers.has(group)) newSpeakers.set(group, large.length + newSpeakers.size);
    target.set(label, newSpeakers.get(group));
  }
  return labels.map((label) => target.get(label));
}
function clusterEmbeddings(embeddings, activity, chunkCount) {
  const pairs = chunkCount * LOCAL_SPEAKERS;
  const { active, clean } = activityTotals(activity, chunkCount);
  const train = [];
  for (let pair = 0; pair < pairs; pair++) {
    if (clean[pair] < MIN_ACTIVE_RATIO * FRAMES_PER_CHUNK) continue;
    if (embeddingAt(embeddings, pair).every(Number.isFinite)) train.push(pair);
  }
  const hardClusters = new Int32Array(pairs);
  if (train.length >= 2) {
    const trainEmbeddings = new Float64Array(train.length * EMBEDDING_DIMENSION);
    train.forEach(
      (pair, row) => trainEmbeddings.set(embeddingAt(embeddings, pair), row * EMBEDDING_DIMENSION)
    );
    const trainClusters = clusterTraining(trainEmbeddings, train.length);
    const clusters = trainClusters.reduce((maximum, label) => Math.max(maximum, label), 0) + 1;
    const centroids = Array.from(
      { length: clusters },
      (_, cluster) => clusterCentroid(trainEmbeddings, trainClusters, cluster)
    );
    for (let pair = 0; pair < pairs; pair++) {
      hardClusters[pair] = mostSimilar(embeddingAt(embeddings, pair), centroids);
    }
  }
  for (let pair = 0; pair < pairs; pair++) {
    if (active[pair] === 0) hardClusters[pair] = UNASSIGNED;
  }
  return hardClusters;
}

// src/lib/server/transcription/diarization/reconstruction.ts
var AGGREGATE_EPSILON = 1e-12;
var BINARIZE_THRESHOLD = 0.5;
function globalFrameCount(chunkCount) {
  const end = CHUNK_WINDOW.start + CHUNK_WINDOW.duration + (chunkCount - 1) * CHUNK_WINDOW.step;
  return closestFrame(end + 0.5 * FRAME_WINDOW.duration, FRAME_WINDOW) + 1;
}
function chunkStartFrame(chunk) {
  const chunkStart = CHUNK_WINDOW.start + chunk * CHUNK_WINDOW.step;
  return closestFrame(chunkStart + 0.5 * FRAME_WINDOW.duration, FRAME_WINDOW);
}
function aggregate(scores, chunkCount, classes) {
  const frames = globalFrameCount(chunkCount);
  const output = new Float32Array(frames * classes);
  const overlap = new Float32Array(frames * classes);
  const covered = new Uint8Array(frames * classes);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const startFrame = chunkStartFrame(chunk);
    for (let frame2 = 0; frame2 < FRAMES_PER_CHUNK; frame2++) {
      const target = startFrame + frame2;
      if (target >= frames) break;
      for (let label = 0; label < classes; label++) {
        const score = scores[(chunk * FRAMES_PER_CHUNK + frame2) * classes + label];
        if (Number.isNaN(score)) continue;
        output[target * classes + label] += score;
        overlap[target * classes + label] += 1;
        covered[target * classes + label] = 1;
      }
    }
  }
  for (let index = 0; index < output.length; index++) {
    output[index] = covered[index] ? output[index] / Math.max(overlap[index], AGGREGATE_EPSILON) : 0;
  }
  return { frames, classes, data: output };
}
function speakerCount(activity, chunkCount) {
  const perChunk = new Float32Array(chunkCount * FRAMES_PER_CHUNK);
  for (let frame2 = 0; frame2 < perChunk.length; frame2++) {
    for (let speaker = 0; speaker < LOCAL_SPEAKERS; speaker++) {
      perChunk[frame2] += activity[frame2 * LOCAL_SPEAKERS + speaker];
    }
  }
  const averaged = aggregate(perChunk, chunkCount, 1);
  return Uint8Array.from(averaged.data, (value) => roundHalfEven(value));
}
function clusterActivations(activity, hardClusters, chunkCount) {
  const clusters = hardClusters.reduce((maximum, cluster) => Math.max(maximum, cluster), -1) + 1;
  const frames = globalFrameCount(chunkCount);
  const data = new Float32Array(frames * clusters);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const startFrame = chunkStartFrame(chunk);
    const chunkClusters = hardClusters.subarray(
      chunk * LOCAL_SPEAKERS,
      (chunk + 1) * LOCAL_SPEAKERS
    );
    for (let frame2 = 0; frame2 < FRAMES_PER_CHUNK; frame2++) {
      const target = startFrame + frame2;
      if (target >= frames) break;
      const base = (chunk * FRAMES_PER_CHUNK + frame2) * LOCAL_SPEAKERS;
      for (let speaker = 0; speaker < LOCAL_SPEAKERS; speaker++) {
        const cluster = chunkClusters[speaker];
        if (cluster === UNASSIGNED || chunkClusters.indexOf(cluster) < speaker) continue;
        let active = 0;
        for (let other = speaker; other < LOCAL_SPEAKERS; other++) {
          if (chunkClusters[other] === cluster) active |= activity[base + other];
        }
        data[target * clusters + cluster] += active;
      }
    }
  }
  return { frames, classes: clusters, data };
}
function toDiarization(activations, count) {
  const maxCount = count.reduce((maximum, value) => Math.max(maximum, value), 0);
  const classes = Math.max(activations.classes, maxCount);
  const frames = Math.min(activations.frames, count.length);
  const binary = new Float32Array(frames * classes);
  const order = Array.from({ length: classes }, (_, index) => index);
  for (let frame2 = 0; frame2 < frames; frame2++) {
    const activation = (label) => label < activations.classes ? activations.data[frame2 * activations.classes + label] : 0;
    const ranked = [...order].sort((left, right) => activation(right) - activation(left));
    for (let rank = 0; rank < count[frame2]; rank++) binary[frame2 * classes + ranked[rank]] = 1;
  }
  return { frames, classes, data: binary };
}
function binarizeToTurns(diarization) {
  const turns = [];
  const middle = (frame2) => FRAME_WINDOW.start + frame2 * FRAME_WINDOW.step + 0.5 * FRAME_WINDOW.duration;
  if (diarization.frames === 0) return turns;
  for (let speaker = 0; speaker < diarization.classes; speaker++) {
    const value = (frame2) => diarization.data[frame2 * diarization.classes + speaker];
    let start = middle(0);
    let active = value(0) > BINARIZE_THRESHOLD;
    for (let frame2 = 1; frame2 < diarization.frames; frame2++) {
      if (active && value(frame2) < BINARIZE_THRESHOLD) {
        turns.push({ start, end: middle(frame2), speaker });
        active = false;
      } else if (!active && value(frame2) > BINARIZE_THRESHOLD) {
        start = middle(frame2);
        active = true;
      }
    }
    if (active) turns.push({ start, end: middle(diarization.frames - 1), speaker });
  }
  return turns.sort(
    (left, right) => left.start - right.start || left.end - right.end || left.speaker - right.speaker
  );
}

// src/lib/server/transcription/diarization/pipeline.ts
async function createSpeakerDiarizer() {
  const models = await loadDiarizationModels();
  const segmentationSession = await InferenceSession.create(models.segmentation);
  const embeddingSession = await InferenceSession.create(models.embedding);
  const diarize = async (samples, onProgress) => {
    const { chunkCount, activity } = await runSegmentation(segmentationSession, samples);
    const count = speakerCount(activity, chunkCount);
    onProgress?.("segmentation", 1);
    if (count.every((value) => value === 0)) return [];
    const embeddings = await extractEmbeddings(
      embeddingSession,
      samples,
      activity,
      chunkCount,
      (fraction) => onProgress?.("embeddings", fraction)
    );
    const hardClusters = clusterEmbeddings(embeddings, activity, chunkCount);
    onProgress?.("clustering", 1);
    const activations = clusterActivations(activity, hardClusters, chunkCount);
    const turns = binarizeToTurns(toDiarization(activations, count));
    onProgress?.("reconstruction", 1);
    return turns;
  };
  return { diarize };
}

// src/lib/server/transcription/forced-alignment.ts
import { AutoModelForCTC, Wav2Vec2Processor } from "@huggingface/transformers";

// src/lib/server/logging/logger.ts
import { homedir } from "node:os";

// src/lib/server/logging/log-file.ts
import { createWriteStream } from "node:fs";
import { mkdir as mkdir2, readFile as readFile2, rename as rename2, stat } from "node:fs/promises";
import { join, resolve as resolve2 } from "node:path";
var LOG_DIRECTORY = resolve2(process.cwd(), "logs");
var LOG_FILE = join(LOG_DIRECTORY, "app.jsonl");
var ROTATED_LOG_FILE = join(LOG_DIRECTORY, "app.1.jsonl");
var MAX_FILE_BYTES = 5 * 1024 * 1024;
var MAX_PENDING_LINES = 1e3;
var stream = null;
var pending = [];
var bytesWritten = 0;
var disabled = false;
function writeLogEntry(entry) {
  if (disabled) return;
  const line = `${JSON.stringify(entry)}
`;
  if (!stream) {
    pending.push(line);
    if (pending.length > MAX_PENDING_LINES) pending.shift();
    return;
  }
  stream.write(line);
  bytesWritten += Buffer.byteLength(line);
  if (bytesWritten >= MAX_FILE_BYTES) void rotateOpenFile();
}
function openStream() {
  const next = createWriteStream(LOG_FILE, { flags: "a" });
  next.on("error", disableLogFile);
  stream = next;
  const queued = pending;
  pending = [];
  for (const line of queued) {
    next.write(line);
    bytesWritten += Buffer.byteLength(line);
  }
}
async function rotateOpenFile() {
  const current = stream;
  if (!current) return;
  stream = null;
  await new Promise((done) => current.end(done));
  try {
    await rotateClosedFile();
    openStream();
  } catch (error) {
    disableLogFile(error);
  }
}
async function rotateClosedFile() {
  await rename2(LOG_FILE, ROTATED_LOG_FILE);
  bytesWritten = 0;
}
function disableLogFile(error) {
  if (disabled) return;
  disabled = true;
  stream = null;
  pending = [];
  console.error("[Logging] Writing logs/app.jsonl failed; file logging is off.", error);
}

// src/lib/server/logging/logger.ts
var MAX_ENTRIES = 1e3;
var PATH_ROOTS = [
  [process.cwd(), "<data>"],
  [homedir(), "~"]
].filter(([root]) => root.length > 1);
var CONSOLE_WRITERS = {
  debug: console.debug,
  info: console.info,
  warn: console.warn,
  error: console.error
};
var runId = crypto.randomUUID();
var entries = [];
var latestId = 0;
function createLogger(subsystem) {
  const method = (level) => (message, fields = {}, cause) => record({ fields, level, message, subsystem, time: (/* @__PURE__ */ new Date()).toISOString() }, cause);
  return {
    debug: method("debug"),
    info: method("info"),
    warn: method("warn"),
    error: method("error")
  };
}
var log = createLogger("application");
function formatLogLine(entry) {
  const fields = Object.entries(entry.fields).map(([key, value]) => `${key}=${String(value)}`);
  const error = entry.error ? [`(${entry.error.name}: ${entry.error.message})`] : [];
  return [
    entry.time,
    entry.level.toUpperCase().padEnd(5),
    entry.subsystem.padEnd(11),
    entry.message,
    ...fields,
    ...error
  ].join(" ");
}
function record(base, cause) {
  const entry = cause === void 0 ? base : { ...base, error: serializeError(cause) };
  remember(entry);
  writeLogEntry(entry);
  const line = formatLogLine(entry);
  const stack = cause instanceof Error ? cause.stack : void 0;
  CONSOLE_WRITERS[entry.level](stack ? `${line}
${stack}` : line);
}
function remember(entry) {
  latestId += 1;
  entries.push({ ...entry, id: latestId });
  if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
}
function serializeError(cause) {
  const scrub = (text) => PATH_ROOTS.reduce((result, [root, label]) => result.replaceAll(root, label), text);
  if (cause instanceof Error) return { message: scrub(cause.message), name: cause.name };
  return { message: scrub(String(cause)), name: "Error" };
}

// src/lib/server/transcription/audio-types.ts
var AUDIO_SAMPLE_RATE = 16e3;
function sampleIndexToMs(sampleIndex) {
  return Math.round(sampleIndex / AUDIO_SAMPLE_RATE * 1e3);
}
function sliceAudioChunk(audioData, startSample, endSample) {
  const start = Math.max(0, Math.min(audioData.length, Math.trunc(startSample)));
  const end = Math.max(start, Math.min(audioData.length, Math.trunc(endSample)));
  return {
    startSample: start,
    endSample: end,
    samples: audioData.subarray(start, end)
  };
}

// src/lib/server/utils/transformers-env.ts
import { resolve as resolve3 } from "node:path";
import { env } from "@huggingface/transformers";
var TRANSFORMERS_CACHE_DIR = resolve3(process.cwd(), ".cache", "transformersjs");
env.allowRemoteModels = true;
env.cacheDir = TRANSFORMERS_CACHE_DIR;
env.localModelPath = TRANSFORMERS_CACHE_DIR;

// src/lib/server/transcription/forced-alignment.ts
var log2 = createLogger("documents");
var ALIGNMENT_MODEL = "Xenova/wav2vec2-base-960h";
var ALIGNMENT_REVISION = "a19f851b3d42865797e410752b4c570c871e4825";
async function loadAlignmentAssets() {
  const [processor, model] = await Promise.all([
    Wav2Vec2Processor.from_pretrained(ALIGNMENT_MODEL, {
      cache_dir: TRANSFORMERS_CACHE_DIR,
      revision: ALIGNMENT_REVISION
    }),
    AutoModelForCTC.from_pretrained(ALIGNMENT_MODEL, {
      cache_dir: TRANSFORMERS_CACHE_DIR,
      device: "cpu",
      dtype: "fp32",
      revision: ALIGNMENT_REVISION
    })
  ]);
  const tokenizer = processor.tokenizer;
  if (!tokenizer) throw new Error(`${ALIGNMENT_MODEL} did not provide a tokenizer.`);
  const vocabulary = tokenizer.get_vocab();
  const blankTokenId = vocabulary.get("<pad>") ?? tokenizer.pad_token_id;
  if (!Number.isInteger(blankTokenId)) {
    throw new Error(`${ALIGNMENT_MODEL} did not provide a CTC blank token.`);
  }
  return { processor, model, vocabulary, blankTokenId };
}
var alignmentAssets;
function getAlignmentAssets() {
  alignmentAssets ??= loadAlignmentAssets().catch((error) => {
    alignmentAssets = void 0;
    throw error;
  });
  return alignmentAssets;
}
function createAlignmentTarget(words, vocabulary) {
  const delimiterTokenId = vocabulary.get("|");
  const target = [];
  words.forEach((word, wordIndex) => {
    const tokens = [];
    const normalized = word.text.replace(/[’‘]/g, "'").normalize("NFKD").replace(new RegExp("\\p{Mark}", "gu"), "");
    for (const character of normalized) {
      const tokenId = vocabulary.get(character) ?? vocabulary.get(character.toUpperCase());
      if (tokenId !== void 0) tokens.push({ tokenId, wordIndex });
      else if (/[\p{Letter}\p{Number}]/u.test(character)) tokens.push({ tokenId: null, wordIndex });
    }
    if (tokens.length === 0) return;
    if (target.length > 0 && delimiterTokenId !== void 0) {
      target.push({ tokenId: delimiterTokenId });
    }
    target.push(...tokens);
  });
  return target;
}
function alignWordFrames(logits, frameCount, vocabularySize, blankTokenId, target, wordCount) {
  const normalizers = new Float64Array(frameCount);
  const blankScores = new Float64Array(frameCount);
  const wildcardScores = new Float64Array(frameCount);
  for (let frame2 = 0; frame2 < frameCount; frame2++) {
    const row = logits.subarray(frame2 * vocabularySize, (frame2 + 1) * vocabularySize);
    let maximum = -Infinity;
    let maximumNonBlank = -Infinity;
    row.forEach((value, tokenId) => {
      maximum = Math.max(maximum, value);
      if (tokenId !== blankTokenId) maximumNonBlank = Math.max(maximumNonBlank, value);
    });
    let exponentialSum = 0;
    for (const value of row) exponentialSum += Math.exp(value - maximum);
    normalizers[frame2] = maximum + Math.log(exponentialSum);
    blankScores[frame2] = row[blankTokenId] - normalizers[frame2];
    wildcardScores[frame2] = maximumNonBlank - normalizers[frame2];
  }
  const tokenScore = (frame2, { tokenId }) => tokenId === null ? wildcardScores[frame2] : logits[frame2 * vocabularySize + tokenId] - normalizers[frame2];
  const columns = target.length + 1;
  const trellis = new Float64Array((frameCount + 1) * columns).fill(-Infinity);
  trellis[0] = 0;
  for (let frame2 = 0; frame2 < frameCount; frame2++) {
    const row = frame2 * columns;
    const next = row + columns;
    trellis[next] = trellis[row] + blankScores[frame2];
    for (let token = 0; token < Math.min(target.length, frame2 + 1); token++) {
      const stayed = trellis[row + token + 1] + blankScores[frame2];
      const advanced = trellis[row + token] + tokenScore(frame2, target[token]);
      trellis[next + token + 1] = Math.max(stayed, advanced);
    }
  }
  let time = -1;
  let bestScore = -Infinity;
  for (let candidate = target.length; candidate <= frameCount; candidate++) {
    const score = trellis[candidate * columns + target.length];
    if (score > bestScore) {
      bestScore = score;
      time = candidate;
    }
  }
  if (time < 0 || !Number.isFinite(bestScore)) return null;
  const spans = new Array(wordCount);
  let column = target.length;
  while (time > 0 && column > 0) {
    const frame2 = time - 1;
    const token = target[column - 1];
    if (token.wordIndex !== void 0) {
      const span = spans[token.wordIndex];
      spans[token.wordIndex] = span ? [Math.min(span[0], frame2), Math.max(span[1], frame2 + 1)] : [frame2, frame2 + 1];
    }
    const stayed = trellis[frame2 * columns + column] + blankScores[frame2];
    const advanced = trellis[frame2 * columns + column - 1] + tokenScore(frame2, token);
    if (advanced > stayed) column--;
    time--;
  }
  return column === 0 ? spans : null;
}
async function alignChunk(chunk, assets) {
  if (chunk.segments.length === 0 || chunk.audio.samples.length === 0) return chunk.segments;
  const target = createAlignmentTarget(chunk.segments, assets.vocabulary);
  if (target.length === 0) return chunk.segments;
  const { logits } = await assets.model(await assets.processor(chunk.audio.samples));
  const [batch, frameCount, vocabularySize] = logits.dims;
  if (logits.dims.length !== 3 || batch !== 1 || frameCount <= 0 || vocabularySize <= assets.blankTokenId || logits.data.length !== frameCount * vocabularySize) {
    throw new Error(`Unexpected CTC logits shape: ${logits.dims.join("x")}`);
  }
  if (target.length > frameCount) return chunk.segments;
  const spans = alignWordFrames(
    logits.data,
    frameCount,
    vocabularySize,
    assets.blankTokenId,
    target,
    chunk.segments.length
  );
  if (!spans) return chunk.segments;
  const sourceStartMs = sampleIndexToMs(chunk.audio.startSample);
  const sourceEndMs = sampleIndexToMs(chunk.audio.endSample);
  const frameDurationMs = chunk.audio.samples.length / AUDIO_SAMPLE_RATE * 1e3 / frameCount;
  const toMs = (frame2) => Math.round(sourceStartMs + frame2 * frameDurationMs);
  return chunk.segments.map((word, wordIndex) => {
    const span = spans[wordIndex];
    if (!span) return word;
    const startMs = Math.min(sourceEndMs, Math.max(sourceStartMs, toMs(span[0])));
    const endMs = Math.min(sourceEndMs, Math.max(startMs, toMs(span[1])));
    return { ...word, startMs, endMs };
  });
}
async function alignTranscription(transcription, onChunk) {
  if (transcription.chunks.length === 0) return transcription;
  let assets;
  try {
    assets = await getAlignmentAssets();
  } catch (error) {
    log2.warn("Forced-alignment model unavailable", {}, error);
    return transcription;
  }
  const alignedChunks = [];
  for (const [index, chunk] of transcription.chunks.entries()) {
    try {
      alignedChunks.push({ ...chunk, segments: await alignChunk(chunk, assets) });
    } catch (error) {
      log2.warn("Could not align an audio chunk", { chunk: index + 1 }, error);
      alignedChunks.push(chunk);
    }
    onChunk?.(index + 1, transcription.chunks.length);
  }
  return {
    text: transcription.text,
    segments: alignedChunks.flatMap((chunk) => chunk.segments),
    chunks: alignedChunks
  };
}

// src/lib/server/transcription/hallucination-gate.ts
import { gzipSync } from "node:zlib";

// src/lib/server/transcription/transcription-model.ts
import {
  ModelRegistry,
  pipeline
} from "@huggingface/transformers";

// src/lib/server/utils/inference-threads.ts
import { availableParallelism } from "node:os";
var INFERENCE_THREADS = Math.max(1, Math.min(8, Math.floor(availableParallelism() / 4)));
var TRANSCRIPTION_THREADS = Math.max(
  1,
  Math.min(8, Math.floor(availableParallelism() / 2))
);

// src/lib/server/transcription/transcription-model.ts
var TRANSCRIPTION_MODEL = "Xenova/whisper-small.en";
var TRANSCRIPTION_TASK = "automatic-speech-recognition";
var TRANSCRIPTION_OPTIONS = {
  cache_dir: TRANSFORMERS_CACHE_DIR,
  device: "cpu",
  dtype: "q8"
};
var transcriptionPipeline;
async function getTranscriber() {
  transcriptionPipeline ??= pipeline(TRANSCRIPTION_TASK, TRANSCRIPTION_MODEL, {
    ...TRANSCRIPTION_OPTIONS,
    session_options: {
      intraOpNumThreads: TRANSCRIPTION_THREADS,
      interOpNumThreads: 1
    }
  }).catch((error) => {
    transcriptionPipeline = void 0;
    throw error;
  });
  return transcriptionPipeline;
}
function decodingOptions(audio) {
  const durationSeconds = audio.samples.length / AUDIO_SAMPLE_RATE;
  const maxNewTokens = Math.min(384, Math.max(32, Math.ceil(durationSeconds * 8)));
  return {
    do_sample: false,
    max_new_tokens: maxNewTokens,
    no_repeat_ngram_size: 4,
    repetition_penalty: 1.05,
    return_timestamps: true
  };
}
function splitSegmentIntoWords(text, startMs, endMs) {
  const words = text.split(/\s+/).filter(Boolean);
  const totalCharacters = words.reduce((sum, word) => sum + word.length, 0);
  const spanMs = endMs - startMs;
  let characters = 0;
  return words.map((word) => {
    const wordStartMs = startMs + Math.round(spanMs * characters / totalCharacters);
    characters += word.length;
    const wordEndMs = startMs + Math.round(spanMs * characters / totalCharacters);
    return { startMs: wordStartMs, endMs: wordEndMs, text: word };
  });
}
async function transcribeChunk(transcriber, audio) {
  const result = await transcriber(audio.samples, decodingOptions(audio));
  const sourceStartMs = sampleIndexToMs(audio.startSample);
  const sourceEndMs = sampleIndexToMs(audio.endSample);
  const segments = (result.chunks ?? []).flatMap((segment) => {
    const text2 = segment.text?.trim() ?? "";
    const [start, end] = segment.timestamp ?? [];
    if (!text2 || typeof start !== "number") return [];
    const startMs = Math.min(
      sourceEndMs,
      Math.max(sourceStartMs, sourceStartMs + Math.round(start * 1e3))
    );
    let endMs = sourceEndMs;
    if (typeof end === "number") {
      endMs = Math.min(sourceEndMs, Math.max(startMs, sourceStartMs + Math.round(end * 1e3)));
    }
    return splitSegmentIntoWords(text2, startMs, endMs);
  });
  let text = result.text.trim();
  if (segments.length > 0) {
    text = segments.map((segment) => segment.text).join(" ");
  }
  return {
    audio,
    text,
    segments
  };
}
function combineChunkResults(chunks) {
  return {
    text: chunks.map((chunk) => chunk.text).filter(Boolean).join(" ").trim(),
    segments: chunks.flatMap((chunk) => chunk.segments),
    chunks
  };
}
async function transcribeAudioChunks(audioChunks, onChunk) {
  if (audioChunks.length === 0) {
    return {
      text: "",
      segments: [],
      chunks: []
    };
  }
  const transcriber = await getTranscriber();
  const results = [];
  for (const [index, audioChunk] of audioChunks.entries()) {
    if (audioChunk.samples.length > 0) {
      results.push(await transcribeChunk(transcriber, audioChunk));
    }
    onChunk?.(index + 1, audioChunks.length);
  }
  return combineChunkResults(results);
}

// src/lib/server/transcription/hallucination-gate.ts
var log3 = createLogger("documents");
var MINIMUM_ANALYSIS_WORDS = 12;
var MAXIMUM_CONSECUTIVE_WORD_RUN = 8;
var MAXIMUM_TRIGRAM_COVERAGE = 0.45;
var MAXIMUM_WORDS_PER_SECOND = 5.5;
var SUSPICIOUS_COMPRESSION_RATIO = 2.4;
function longestConsecutiveWordRun(words) {
  let longest = 0;
  let current = 0;
  words.forEach((word, index) => {
    current = word === words[index - 1] ? current + 1 : 1;
    longest = Math.max(longest, current);
  });
  return longest;
}
function trigramCoverage(words) {
  const counts = /* @__PURE__ */ new Map();
  let greatestCount = 0;
  for (let index = 0; index + 3 <= words.length; index++) {
    const key = words.slice(index, index + 3).join("\0");
    const count = (counts.get(key) ?? 0) + 1;
    counts.set(key, count);
    greatestCount = Math.max(greatestCount, count);
  }
  return Math.min(1, greatestCount * 3 / words.length);
}
function compressionRatio(text) {
  const byteLength = Buffer.byteLength(text, "utf8");
  if (byteLength < 100) return 0;
  return byteLength / gzipSync(text).byteLength;
}
function hallucinationReasons(chunk) {
  const text = chunk.text.trim();
  const words = text.toLowerCase().match(/[\p{Letter}\p{Number}']+/gu) ?? [];
  if (words.length < MINIMUM_ANALYSIS_WORDS) return [];
  const reasons = [];
  const run = longestConsecutiveWordRun(words);
  if (run >= MAXIMUM_CONSECUTIVE_WORD_RUN) reasons.push(`${run} consecutive copies of one word`);
  const coverage = trigramCoverage(words);
  if (words.length >= 24 && coverage >= MAXIMUM_TRIGRAM_COVERAGE) {
    reasons.push(`${Math.round(coverage * 100)}% repeated-trigram coverage`);
  }
  const seconds = Math.max(chunk.audio.samples.length / AUDIO_SAMPLE_RATE, Number.EPSILON);
  const wordsPerSecond = words.length / seconds;
  const ratio = compressionRatio(text);
  if (words.length >= 30 && wordsPerSecond >= MAXIMUM_WORDS_PER_SECOND && ratio >= SUSPICIOUS_COMPRESSION_RATIO) {
    reasons.push(
      `${wordsPerSecond.toFixed(1)} words/second with compression ratio ${ratio.toFixed(2)}`
    );
  }
  return reasons;
}
function filterHallucinatedTranscription(transcription) {
  const accepted = transcription.chunks.filter((chunk, index) => {
    const reasons = hallucinationReasons(chunk);
    if (reasons.length === 0) return true;
    log3.warn("Rejected probable hallucination", {
      chunk: index + 1,
      endMs: sampleIndexToMs(chunk.audio.endSample),
      reasons: reasons.join(", "),
      startMs: sampleIndexToMs(chunk.audio.startSample)
    });
    return false;
  });
  return combineChunkResults(accepted);
}

// src/lib/server/transcription/speaker-assignment.ts
var NEAREST_TURN_MAX_GAP_MS = 500;
var overlapBySpeaker = /* @__PURE__ */ new Map();
function overlappingSpeaker(word, turns) {
  overlapBySpeaker.clear();
  for (const turn of turns) {
    const overlapMs = Math.min(word.endMs, turn.end * 1e3) - Math.max(word.startMs, turn.start * 1e3);
    if (overlapMs <= 0) continue;
    overlapBySpeaker.set(turn.speaker, (overlapBySpeaker.get(turn.speaker) ?? 0) + overlapMs);
  }
  let speakerId;
  let greatestOverlapMs = 0;
  for (const [speaker, overlapMs] of overlapBySpeaker) {
    if (overlapMs <= greatestOverlapMs) continue;
    speakerId = speaker;
    greatestOverlapMs = overlapMs;
  }
  return speakerId;
}
function nearestTurnSpeaker(word, turns) {
  if (!/[\p{Letter}\p{Number}]/u.test(word.text)) return void 0;
  let speakerId;
  let smallestGapMs = NEAREST_TURN_MAX_GAP_MS;
  for (const turn of turns) {
    const gapMs = Math.max(turn.start * 1e3 - word.endMs, word.startMs - turn.end * 1e3, 0);
    if (gapMs > smallestGapMs) continue;
    if (speakerId !== void 0 && gapMs === smallestGapMs) continue;
    speakerId = turn.speaker;
    smallestGapMs = gapMs;
  }
  return speakerId;
}
function assignSpeakers(words, turns) {
  const orderedTurns = turns.toSorted((left, right) => left.start - right.start);
  const wordOrder = words.map((_, index) => index).sort((left, right) => words[left].startMs - words[right].startMs);
  const assigned = words.slice();
  const window2 = [];
  let nextTurn = 0;
  for (const index of wordOrder) {
    const word = words[index];
    while (nextTurn < orderedTurns.length && orderedTurns[nextTurn].start * 1e3 - word.endMs <= NEAREST_TURN_MAX_GAP_MS) {
      window2.push(orderedTurns[nextTurn++]);
    }
    let kept = 0;
    for (const turn of window2) {
      if (word.startMs - turn.end * 1e3 <= NEAREST_TURN_MAX_GAP_MS) window2[kept++] = turn;
    }
    window2.length = kept;
    const speakerId = overlappingSpeaker(word, window2) ?? nearestTurnSpeaker(word, window2);
    if (speakerId !== void 0) assigned[index] = { ...word, speakerId };
  }
  return assigned;
}

// src/lib/server/transcription/voice-activity-detection.ts
import { existsSync } from "node:fs";
import { resolve as resolve4 } from "node:path";
import { InferenceSession as InferenceSession2, Tensor as Tensor3 } from "onnxruntime-node";
var VAD_MODEL_PATH = resolve4(process.cwd(), "models", "vad", "silero_vad_v4.onnx");
var VAD_WINDOW_SAMPLES = 512;
var VAD_STATE_DIMS = [2, 1, 64];
var VAD_STATE_SIZE = 2 * 64;
var fromMilliseconds = (milliseconds) => AUDIO_SAMPLE_RATE * milliseconds / 1e3;
var SPEECH_THRESHOLD = 0.5;
var SILENCE_THRESHOLD = Math.max(SPEECH_THRESHOLD - 0.15, 0.01);
var MIN_SPEECH_SAMPLES = fromMilliseconds(250);
var MIN_SILENCE_SAMPLES = fromMilliseconds(500);
var SPEECH_PAD_SAMPLES = fromMilliseconds(30);
var MAX_SPEECH_SAMPLES = fromMilliseconds(25e3) - VAD_WINDOW_SAMPLES - 2 * SPEECH_PAD_SAMPLES;
var MIN_SILENCE_SAMPLES_AT_MAX_SPEECH = fromMilliseconds(98);
var MAX_CHUNK_SAMPLES = 28 * AUDIO_SAMPLE_RATE;
var MAX_MERGE_GAP_SAMPLES = Math.round(0.75 * AUDIO_SAMPLE_RATE);
var CHUNK_PADDING_SAMPLES = Math.round(0.25 * AUDIO_SAMPLE_RATE);
var vadSession;
function getVadSession() {
  vadSession ??= InferenceSession2.create(VAD_MODEL_PATH, {
    intraOpNumThreads: 1,
    interOpNumThreads: 1
  }).catch((error) => {
    vadSession = void 0;
    throw error;
  });
  return vadSession;
}
async function speechProbabilities(session, audioData) {
  const probabilities = new Float32Array(Math.ceil(audioData.length / VAD_WINDOW_SAMPLES));
  const sampleRate = new Tensor3("int64", BigInt64Array.of(BigInt(AUDIO_SAMPLE_RATE)), []);
  const window2 = new Float32Array(VAD_WINDOW_SAMPLES);
  let h = new Float32Array(VAD_STATE_SIZE);
  let c = new Float32Array(VAD_STATE_SIZE);
  for (let index = 0; index < probabilities.length; index++) {
    const samples = audioData.subarray(
      index * VAD_WINDOW_SAMPLES,
      (index + 1) * VAD_WINDOW_SAMPLES
    );
    window2.set(samples);
    window2.fill(0, samples.length);
    const output = await session.run({
      input: new Tensor3("float32", window2, [1, VAD_WINDOW_SAMPLES]),
      sr: sampleRate,
      h: new Tensor3("float32", h, VAD_STATE_DIMS),
      c: new Tensor3("float32", c, VAD_STATE_DIMS)
    });
    h = output.hn.data;
    c = output.cn.data;
    probabilities[index] = output.output.data[0];
  }
  return probabilities;
}
function speechRanges(probabilities, sampleCount) {
  const ranges = [];
  let triggered = false;
  let start;
  let tempEnd = 0;
  let possibleEnds = [];
  for (let index = 0; index < probabilities.length; index++) {
    const probability = probabilities[index];
    const sample = VAD_WINDOW_SAMPLES * index;
    if (probability >= SPEECH_THRESHOLD && tempEnd) {
      const silence = sample - tempEnd;
      if (silence > MIN_SILENCE_SAMPLES_AT_MAX_SPEECH) possibleEnds.push([tempEnd, silence]);
      tempEnd = 0;
    }
    if (probability >= SPEECH_THRESHOLD && !triggered) {
      triggered = true;
      start = sample;
      continue;
    }
    if (triggered && start !== void 0 && sample - start > MAX_SPEECH_SAMPLES) {
      if (possibleEnds.length > 0) {
        const [end, silence] = possibleEnds.reduce(
          (longest, candidate) => candidate[1] > longest[1] ? candidate : longest
        );
        ranges.push({ startSample: start, endSample: end });
        start = void 0;
        if (end + silence < end + sample) start = end + silence;
        else triggered = false;
        tempEnd = 0;
        possibleEnds = [];
      } else {
        ranges.push({ startSample: start, endSample: sample });
        start = void 0;
        tempEnd = 0;
        triggered = false;
        possibleEnds = [];
        continue;
      }
    }
    if (probability < SILENCE_THRESHOLD && triggered) {
      if (!tempEnd) tempEnd = sample;
      if (sample - tempEnd < MIN_SILENCE_SAMPLES) continue;
      if (start !== void 0 && tempEnd - start > MIN_SPEECH_SAMPLES) {
        ranges.push({ startSample: start, endSample: tempEnd });
      }
      start = void 0;
      tempEnd = 0;
      triggered = false;
      possibleEnds = [];
    }
  }
  if (start !== void 0 && sampleCount - start > MIN_SPEECH_SAMPLES) {
    ranges.push({ startSample: start, endSample: sampleCount });
  }
  return padSpeechRanges(ranges, sampleCount);
}
function padSpeechRanges(ranges, sampleCount) {
  ranges.forEach((range, index) => {
    if (index === 0) range.startSample = Math.max(0, range.startSample - SPEECH_PAD_SAMPLES);
    const next = ranges[index + 1];
    if (!next) {
      range.endSample = Math.min(sampleCount, range.endSample + SPEECH_PAD_SAMPLES);
      return;
    }
    const silence = next.startSample - range.endSample;
    if (silence < 2 * SPEECH_PAD_SAMPLES) {
      range.endSample += Math.floor(silence / 2);
      next.startSample = Math.max(0, next.startSample - Math.floor(silence / 2));
    } else {
      range.endSample = Math.min(sampleCount, range.endSample + SPEECH_PAD_SAMPLES);
      next.startSample = Math.max(0, next.startSample - SPEECH_PAD_SAMPLES);
    }
  });
  return ranges;
}
function splitLongRanges(ranges) {
  const splitRanges = [];
  for (const range of ranges) {
    for (let startSample = range.startSample; startSample < range.endSample; startSample += MAX_CHUNK_SAMPLES) {
      splitRanges.push({
        startSample,
        endSample: Math.min(range.endSample, startSample + MAX_CHUNK_SAMPLES)
      });
    }
  }
  return splitRanges;
}
function mergeSpeechRanges(ranges) {
  const mergedRanges = [];
  for (const range of splitLongRanges(ranges)) {
    const previous = mergedRanges.at(-1);
    if (!previous) {
      mergedRanges.push({ ...range });
      continue;
    }
    const gapSamples = range.startSample - previous.endSample;
    const combinedEndSample = Math.max(previous.endSample, range.endSample);
    const combinedLength = combinedEndSample - previous.startSample;
    if (gapSamples <= MAX_MERGE_GAP_SAMPLES && combinedLength <= MAX_CHUNK_SAMPLES) {
      previous.endSample = combinedEndSample;
      continue;
    }
    mergedRanges.push({ ...range });
  }
  return mergedRanges;
}
function createPaddedChunks(audioData, ranges) {
  return ranges.map((range, index) => {
    const previous = ranges[index - 1];
    const next = ranges[index + 1];
    let leftBoundary = 0;
    if (previous) {
      leftBoundary = Math.floor((previous.endSample + range.startSample) / 2);
    }
    let rightBoundary = audioData.length;
    if (next) {
      rightBoundary = Math.floor((range.endSample + next.startSample) / 2);
    }
    const startSample = Math.max(leftBoundary, range.startSample - CHUNK_PADDING_SAMPLES);
    const endSample = Math.min(rightBoundary, range.endSample + CHUNK_PADDING_SAMPLES);
    return sliceAudioChunk(audioData, startSample, endSample);
  });
}
async function detectSpeechChunks(audioData) {
  if (!existsSync(VAD_MODEL_PATH)) {
    throw new Error(`Silero VAD model is missing from ${VAD_MODEL_PATH}`);
  }
  const probabilities = await speechProbabilities(await getVadSession(), audioData);
  const mergedRanges = mergeSpeechRanges(speechRanges(probabilities, audioData.length));
  return createPaddedChunks(audioData, mergedRanges);
}

// src/lib/server/transcription/transcription-worker-entry.ts
if (!parentPort) throw new Error("The transcription worker must run in a worker thread.");
var port = parentPort;
var diarizer;
function loadDiarizer() {
  diarizer ??= createSpeakerDiarizer().catch((error) => {
    diarizer = void 0;
    throw error;
  });
  return diarizer;
}
function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
function reply(message) {
  port.postMessage(message);
}
async function transcribe({ id, samples }) {
  const progress = (stage, fraction) => reply({ type: "progress", id, stage, fraction });
  try {
    progress("speech", 0);
    const audioChunks = await detectSpeechChunks(samples);
    if (audioChunks.length === 0) {
      reply({ type: "result", id, text: "", segments: [] });
      return;
    }
    progress("transcription", 0);
    let transcription = filterHallucinatedTranscription(
      await transcribeAudioChunks(
        audioChunks,
        (completed, total) => progress("transcription", completed / total)
      )
    );
    if (transcription.segments.length > 0) {
      progress("alignment", 0);
      transcription = await alignTranscription(
        transcription,
        (completed, total) => progress("alignment", completed / total)
      );
    }
    let segments = transcription.segments;
    let speakerError;
    if (segments.length > 0) {
      progress("segmentation", 0);
      try {
        const turns = await (await loadDiarizer()).diarize(samples, progress);
        segments = assignSpeakers(segments, turns);
      } catch (error) {
        speakerError = errorMessage(error);
      }
    }
    reply({ type: "result", id, text: transcription.text, segments, speakerError });
  } catch (error) {
    reply({ type: "error", id, message: errorMessage(error) });
  }
}
port.on("message", (request) => {
  if (request.type === "transcribe") void transcribe(request);
});
//# sourceMappingURL=transcription-worker.mjs.map
