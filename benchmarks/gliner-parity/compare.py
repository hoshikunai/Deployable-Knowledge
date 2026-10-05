from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity"
REFERENCE = "torch"
CANDIDATES = ("onnx-fp32", "onnx-int8")
FP32_MAX_SCORE_DIFF = 0.01
INT8_MIN_F1 = 0.95


def load(variant: str) -> tuple[dict[str, Any], Any]:
    result = json.loads((OUT / f"{variant}.json").read_text(encoding="utf-8"))
    return result, np.load(OUT / f"{variant}-logits.npz")


def spans(items: list[dict[str, Any]]) -> dict[tuple[int, int, str], float]:
    return {(item["start"], item["end"], item["label"]): item["score"] for item in items}


def agreement(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    expected, found = spans(reference), spans(candidate)
    shared = expected.keys() & found.keys()
    precision = len(shared) / len(found) if found else 1.0
    recall = len(shared) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    diffs = [abs(expected[key] - found[key]) for key in shared]
    return {"reference": len(expected), "candidate": len(found), "shared": len(shared), "precision": precision,
            "recall": recall, "f1": f1, "maxScoreDiff": max(diffs, default=0.0),
            "meanScoreDiff": statistics.fmean(diffs) if diffs else 0.0}


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = sum(row["reference"] for row in rows)
    candidate = sum(row["candidate"] for row in rows)
    shared = sum(row["shared"] for row in rows)
    precision = shared / candidate if candidate else 1.0
    recall = shared / reference if reference else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"reference": reference, "candidate": candidate, "shared": shared, "precision": precision,
            "recall": recall, "f1": f1, "maxScoreDiff": max(row["maxScoreDiff"] for row in rows)}


def logit_diffs(reference: Any, candidate: Any, prefix: str) -> dict[str, float]:
    keys = sorted(key for key in reference.files if key.startswith(prefix) and not key.startswith("ids"))
    maxima, means, probability = [], [], []
    for key in keys:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape:
            raise ValueError(f"{key}: shapes differ, {left.shape} vs {right.shape}")
        delta = np.abs(left - right)
        maxima.append(float(delta.max()))
        means.append(float(delta.mean()))
        probability.append(float(np.abs(1 / (1 + np.exp(-left)) - 1 / (1 + np.exp(-right))).max()))
    return {"maxAbs": max(maxima, default=0.0), "meanAbs": statistics.fmean(means) if means else 0.0,
            "maxProbability": max(probability, default=0.0)}


def timing(result: dict[str, Any]) -> dict[str, float]:
    entity = sorted(chunk["entitySeconds"] for chunk in result["chunks"])
    relation = sorted(chunk["relationSeconds"] for chunk in result["chunks"] if chunk["relationPrompts"])
    words = sum(chunk["words"] for chunk in result["chunks"])
    return {"entityMedian": statistics.median(entity), "entityP95": entity[int(0.95 * (len(entity) - 1))],
            "relationMedian": statistics.median(relation) if relation else 0.0,
            "secondsPerChunk": (sum(entity) + sum(relation)) / len(entity),
            "wordsPerSecond": words / sum(entity), "loadSeconds": result["loadSeconds"],
            "peakRssMb": result["peakRssMb"]}


def verdict(variant: str, entities: dict[str, Any], relations: dict[str, Any]) -> str:
    if variant == "onnx-fp32":
        exact = entities["f1"] == 1.0 and relations["f1"] == 1.0
        close = max(entities["maxScoreDiff"], relations["maxScoreDiff"]) <= FP32_MAX_SCORE_DIFF
        return "pass" if exact and close else "fail"
    return "pass" if min(entities["f1"], relations["f1"]) >= INT8_MIN_F1 else "fail"


def main() -> int:
    reference, reference_logits = load(REFERENCE)
    summary: dict[str, Any] = {"reference": REFERENCE, "timing": {REFERENCE: timing(reference)}, "variants": {}}
    for variant in CANDIDATES:
        if not (OUT / f"{variant}.json").is_file():
            continue
        result, logits = load(variant)
        tokens_match = all(np.array_equal(reference_logits[key], logits[key])
                           for key in reference_logits.files if key.startswith("ids"))
        entity_rows = [agreement(left["entities"], right["entities"])
                       for left, right in zip(reference["chunks"], result["chunks"], strict=True)]
        relation_rows = [agreement(left["relations"], right["relations"])
                         for left, right in zip(reference["chunks"], result["chunks"], strict=True)]
        entities, relations = pooled(entity_rows), pooled(relation_rows)
        summary["timing"][variant] = timing(result)
        summary["variants"][variant] = {
            "tokensMatch": tokens_match,
            "entityLogits": logit_diffs(reference_logits, logits, "e"),
            "relationLogits": logit_diffs(reference_logits, logits, "r"),
            "entities": entities,
            "relations": relations,
            "verdict": verdict(variant, entities, relations),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"| Variant | Tokens match | Max logit diff | Max prob diff | Entity F1 (shared/ref) | "
          f"Relation F1 (shared/ref) | Max score diff | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for variant, row in summary["variants"].items():
        print(f"| {variant} | {row['tokensMatch']} | {row['entityLogits']['maxAbs']:.4f} | "
              f"{row['entityLogits']['maxProbability']:.4f} | {row['entities']['f1']:.3f} "
              f"({row['entities']['shared']}/{row['entities']['reference']}) | {row['relations']['f1']:.3f} "
              f"({row['relations']['shared']}/{row['relations']['reference']}) | "
              f"{max(row['entities']['maxScoreDiff'], row['relations']['maxScoreDiff']):.4f} | {row['verdict']} |")
    print()
    print("| Variant | Load s | Peak MB | Entity pass median s | p95 s | Relation pass median s | "
          "s per chunk | Words/s |")
    print("|---|---|---|---|---|---|---|---|")
    for variant, row in summary["timing"].items():
        print(f"| {variant} | {row['loadSeconds']:.1f} | {row['peakRssMb']:.0f} | {row['entityMedian']:.2f} | "
              f"{row['entityP95']:.2f} | {row['relationMedian']:.2f} | {row['secondsPerChunk']:.2f} | "
              f"{row['wordsPerSecond']:.0f} |")
    return 0 if all(row["verdict"] == "pass" for row in summary["variants"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
