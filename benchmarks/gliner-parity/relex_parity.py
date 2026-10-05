from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import onnxruntime
import torch
from gliner import GLiNER

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity"
RELEX_REPO = "knowledgator/gliner-relex-large-v0.5"
EXPORT_DIR = OUT / "relex-onnx"
THREADS = max(1, min(8, (os.cpu_count() or 4) // 4))
ENTITY_THRESHOLD = 0.5
ADJACENCY_THRESHOLD = 0.5
RELATION_THRESHOLD = 0.7
EMPTY_TEXT = "It rained for most of the afternoon and the streets stayed quiet."


def load(variant: str) -> GLiNER:
    if variant in ("torch", "split"):
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        model = GLiNER.from_pretrained(RELEX_REPO).to("cpu")
        model.eval()
        if variant == "split":
            from relex_split import SplitRunner, onnx_sessions, route_through
            route_through(model, SplitRunner(sessions=onnx_sessions()))
        return model
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    return GLiNER.from_pretrained(str(EXPORT_DIR), runtime="onnxruntime", runtime_model_file="model.onnx",
                                  runtime_options={"session_options": options,
                                                   "providers": ["CPUExecutionProvider"]})


def extract(model: GLiNER, texts: list[str], labels: list[str], relations: list[str]) -> list[dict[str, Any]]:
    entities, found = model.inference(texts=texts, labels=labels, relations=relations,
                                      threshold=ENTITY_THRESHOLD, adjacency_threshold=ADJACENCY_THRESHOLD,
                                      relation_threshold=RELATION_THRESHOLD, return_relations=True, flat_ner=True)
    return [{
        "entities": [{"start": item["start"], "end": item["end"], "text": item["text"], "label": item["label"],
                      "score": float(item["score"])} for item in entities[index]],
        "relations": [{"head": [item["head"]["start"], item["head"]["end"]], "headText": item["head"]["text"],
                       "relation": item["relation"], "tail": [item["tail"]["start"], item["tail"]["end"]],
                       "tailText": item["tail"]["text"], "score": float(item["score"])}
                      for item in found[index]],
    } for index in range(len(texts))]


def attempt(name: str, action) -> dict[str, Any]:
    try:
        return {"case": name, "ok": True, "result": action()}
    except Exception as error:
        return {"case": name, "ok": False, "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc().splitlines()[-6:]}


def run(variant: str) -> None:
    texts = json.loads((OUT / "texts.json").read_text(encoding="utf-8"))
    labels, relations = texts["entityLabels"], texts["relations"]
    started = time.perf_counter()
    model = load(variant)
    load_seconds = time.perf_counter() - started
    attempt("warm-up", lambda: extract(model, [texts["chunks"][0]["text"]], labels, relations))
    chunks = []
    for index, chunk in enumerate(texts["chunks"]):
        began = time.perf_counter()
        outcome = attempt(chunk["id"], lambda: extract(model, [chunk["text"]], labels, relations)[0])
        outcome["seconds"] = time.perf_counter() - began
        chunks.append(outcome)
        summary = (f"{len(outcome['result']['entities'])} entities, {len(outcome['result']['relations'])} relations"
                   if outcome["ok"] else outcome["error"][:120])
        print(f"relex-{variant} {index + 1}/{len(texts['chunks'])}: {summary} ({outcome['seconds']:.2f}s)",
              flush=True)
    edge_cases = [
        attempt("no-entities", lambda: extract(model, [EMPTY_TEXT], labels, relations)[0]),
        attempt("batch-of-two", lambda: extract(model, [texts["chunks"][1]["text"], texts["chunks"][2]["text"]],
                                                labels, relations)),
    ]
    for case in edge_cases:
        print(f"relex-{variant} edge case {case['case']}: {'ok' if case['ok'] else case['error'][:160]}")
    result = {"variant": variant, "threads": THREADS, "loadSeconds": load_seconds,
              "peakRssMb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "chunks": chunks, "edgeCases": edge_cases}
    (OUT / f"relex-{variant}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")


def keyed(items: list[dict[str, Any]], relation: bool) -> dict[tuple, float]:
    if relation:
        return {(tuple(item["head"]), item["relation"], tuple(item["tail"])): item["score"] for item in items}
    return {(item["start"], item["end"], item["label"]): item["score"] for item in items}


def f1(expected: dict, found: dict) -> tuple[float, int, int, float]:
    shared = expected.keys() & found.keys()
    precision = len(shared) / len(found) if found else 1.0
    recall = len(shared) / len(expected) if expected else 1.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return score, len(shared), len(expected), max((abs(expected[k] - found[k]) for k in shared), default=0.0)


def compare(candidate_variant: str) -> int:
    reference = json.loads((OUT / "relex-torch.json").read_text(encoding="utf-8"))
    candidate = json.loads((OUT / f"relex-{candidate_variant}.json").read_text(encoding="utf-8"))
    totals = {"entities": [{}, {}], "relations": [{}, {}]}
    failures = 0
    for left, right in zip(reference["chunks"], candidate["chunks"], strict=True):
        if not (left["ok"] and right["ok"]):
            failures += 1
            continue
        for kind, relation in (("entities", False), ("relations", True)):
            totals[kind][0].update({(left["case"], *key): value
                                    for key, value in keyed(left["result"][kind], relation).items()})
            totals[kind][1].update({(left["case"], *key): value
                                    for key, value in keyed(right["result"][kind], relation).items()})
    compared = len(reference["chunks"]) - failures
    print("| Check | Result |")
    print("|---|---|")
    for kind in ("entities", "relations"):
        if not compared:
            print(f"| {kind} F1 vs torch | not measurable: no chunk succeeded in both runs |")
            continue
        score, shared, expected, diff = f1(*totals[kind])
        print(f"| {kind} F1 vs torch | {score:.3f} ({shared}/{expected}), max score diff {diff:.4f} |")
    print(f"| chunks that failed in either run | {failures} of {len(reference['chunks'])} |")
    for case_left, case_right in zip(reference["edgeCases"], candidate["edgeCases"], strict=True):
        print(f"| edge case {case_left['case']} | torch {'ok' if case_left['ok'] else 'error'}, "
              f"{candidate_variant} {'ok' if case_right['ok'] else case_right['error'][:90]} |")
    for result in (reference, candidate):
        seconds = [chunk["seconds"] for chunk in result["chunks"] if chunk["ok"]]
        median = f"median {statistics.median(seconds):.2f}s per chunk" if seconds else "no successful chunks"
        print(f"| {result['variant']} timing | load {result['loadSeconds']:.1f}s, {median}, "
              f"peak {result['peakRssMb']:.0f} MB |")
    return 0 if compared == len(reference["chunks"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exported GLiNER-relex ONNX with PyTorch.")
    parser.add_argument("action", choices=("torch", "onnx", "split", "compare"))
    parser.add_argument("--candidate", choices=("onnx", "split", "ts"), default="onnx")
    arguments = parser.parse_args()
    if arguments.action == "compare":
        return compare(arguments.candidate)
    run(arguments.action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
