from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import torch
from gliner import GLiNER

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity"
PYTORCH_REPO = "knowledgator/gliner-multitask-large-v0.5"
ONNX_REPO = "onnx-community/gliner-multitask-large-v0.5"
ONNX_FILES = {"onnx-fp32": "onnx/model.onnx", "onnx-int8": "onnx/model_int8.onnx"}
VARIANTS = ("torch", *ONNX_FILES)
THREADS = max(1, min(8, (os.cpu_count() or 4) // 4))
THRESHOLD = 0.5
RELATION_HEADS = 4


def load(variant: str) -> GLiNER:
    if variant == "torch":
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        model = GLiNER.from_pretrained(PYTORCH_REPO).to("cpu")
        model.eval()
        return model
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = GLiNER.from_pretrained(
        ONNX_REPO,
        runtime="onnxruntime",
        runtime_model_file=ONNX_FILES[variant],
        runtime_options={"session_options": options, "providers": ["CPUExecutionProvider"]},
    )
    run_inference = model.model.run_inference

    def classes_last(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = run_inference(inputs)
        outputs["logits"] = np.ascontiguousarray(np.transpose(outputs["logits"], (1, 2, 3, 0)))
        return outputs

    model.model.run_inference = classes_last
    return model


def raw_logits(model: GLiNER, text: str, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    prepared = model.prepare_batch([text], labels)
    batch = model.collate_batch(prepared["input_x"], prepared["entity_types"], label_names=prepared["label_names"])
    output = model.run_batch(batch, threshold=THRESHOLD)
    logits = output.logits
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    return np.asarray(batch["input_ids"]), np.asarray(logits, dtype=np.float32)


def timed_predict(model: GLiNER, text: str, labels: list[str]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    found = model.predict_entities(text, labels, threshold=THRESHOLD, flat_ner=True)
    seconds = time.perf_counter() - started
    entities = [{"start": item["start"], "end": item["end"], "text": item["text"], "label": item["label"],
                 "score": float(item["score"])} for item in found]
    return entities, seconds


def relation_prompts(entities: list[dict[str, Any]], relations: list[str]) -> list[str]:
    heads: list[str] = []
    for entity in sorted(entities, key=lambda item: item["score"], reverse=True):
        if entity["text"] not in heads:
            heads.append(entity["text"])
        if len(heads) == RELATION_HEADS:
            break
    return [f"{head} <> {relation}" for head in heads for relation in relations]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GLiNER multitask variant over the fixed test texts.")
    parser.add_argument("variant", choices=VARIANTS)
    arguments = parser.parse_args()
    texts = json.loads((OUT / "texts.json").read_text(encoding="utf-8"))
    prompts_path = OUT / "prompts.json"
    if arguments.variant != "torch" and not prompts_path.is_file():
        raise FileNotFoundError("Run the torch variant first; it writes the shared relation prompts.")
    shared_prompts = json.loads(prompts_path.read_text(encoding="utf-8")) if prompts_path.is_file() else {}

    started = time.perf_counter()
    model = load(arguments.variant)
    load_seconds = time.perf_counter() - started
    labels = texts["entityLabels"]
    timed_predict(model, texts["chunks"][0]["text"], labels)

    arrays: dict[str, np.ndarray] = {}
    chunks = []
    prompts_out: dict[str, list[str]] = {}
    for index, chunk in enumerate(texts["chunks"]):
        entities, entity_seconds = timed_predict(model, chunk["text"], labels)
        prompts = shared_prompts.get(chunk["id"]) if arguments.variant != "torch" else relation_prompts(
            entities, texts["relations"])
        prompts_out[chunk["id"]] = prompts
        relations, relation_seconds = timed_predict(model, chunk["text"], prompts) if prompts else ([], 0.0)
        arrays[f"ids{index}"], arrays[f"e{index}"] = raw_logits(model, chunk["text"], labels)
        if prompts:
            _, arrays[f"r{index}"] = raw_logits(model, chunk["text"], prompts)
        chunks.append({"id": chunk["id"], "source": chunk["source"], "words": len(chunk["text"].split()),
                       "entities": entities, "entitySeconds": entity_seconds, "relationPrompts": prompts,
                       "relations": relations, "relationSeconds": relation_seconds})
        print(f"{arguments.variant} {index + 1}/{len(texts['chunks'])}: {len(entities)} entities, "
              f"{len(relations)} relations, {entity_seconds:.2f}s + {relation_seconds:.2f}s", flush=True)

    if arguments.variant == "torch":
        prompts_path.write_text(json.dumps(prompts_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / f"{arguments.variant}-logits.npz", **arrays)
    result = {
        "variant": arguments.variant,
        "threads": THREADS,
        "threshold": THRESHOLD,
        "loadSeconds": load_seconds,
        "peakRssMb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "chunks": chunks,
    }
    (OUT / f"{arguments.variant}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"{arguments.variant}: loaded in {load_seconds:.1f}s, peak {result['peakRssMb']:.0f} MB")


if __name__ == "__main__":
    main()
