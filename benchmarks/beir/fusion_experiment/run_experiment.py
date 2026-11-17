"""Run the fusion experiment: RRF weights, fusion level, document ordering, and title input.

1. Tune (RRF only) on SciFact train and NFCorpus dev: five semantic:BM25 weight pairs at chunk
   level, then fusion level x document ordering at the pair with the best average nDCG@10.
2. Report on test, choosing nothing there: SciFact and NFCorpus (all queries) and ArguAna
   (300-query sample; it has no train/dev split). Per dataset: RRF baseline and tuned, and
   RRF + Ettin baseline, with title, tuned, and tuned with title.

Settings are pinned here, not taken from the bench GUI. Run IDs are cached in manifest.json
by configuration; running the script again skips finished runs and resumes unfinished ones.

    benchmarks/beir/.venv/bin/python benchmarks/beir/fusion_experiment/run_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT.parents[1] / "bench-gui"))

import paths  # noqa: E402
from bench import run_job  # noqa: E402
from jobs import Settings, create_jobs, load_job, save_job  # noqa: E402

MANIFEST = EXPERIMENT_ROOT / "manifest.json"
PINNED = {
    "retrievalMultiplier": 5, "rerankMultiplier": 2, "rrfRankConstant": 60, "documentDepth": 10,
    "rerankMaxTokens": 512, "chunkMaxTokens": 0, "chunkOverlapTokens": 0,
    "embeddingModel": "nomic-embed-text-v1.5", "dedupMode": "none", "maxChunksPerDocument": 2,
    "timeoutMinutes": 30, "maxSessions": 8,
}
BASELINE = {"rrfSemanticWeight": 1.0, "rrfBm25Weight": 1.0, "fusionLevel": "chunk",
            "documentOrder": "none", "rerankTitle": "off"}
TUNING = {"scifact": "train", "nfcorpus": "dev"}
WEIGHTS = [(1.0, 1.0), (1.0, 0.75), (1.0, 0.5), (0.75, 1.0), (0.5, 1.0)]
TEST = {"scifact": ("all", 42), "nfcorpus": ("all", 42), "arguana": ("300", 42)}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def config_key(dataset: str, pipeline: str, queries: str, fusion: dict, split: str) -> str:
    return json.dumps([dataset, pipeline, queries, split, sorted(fusion.items())])


def ensure_run(manifest: dict, dataset: str, pipeline: str, queries: str, seed: int,
               fusion: dict, split: str) -> str | None:
    """Run (or resume) one configuration once; return its run ID when complete."""
    key = config_key(dataset, pipeline, queries, fusion, split)
    runs = manifest.setdefault("runs", {})
    run_id = runs.get(key)
    if run_id:
        spec = load_job(run_id)
        if spec.state == "complete":
            return run_id
        print(f"Resuming {run_id}", flush=True)
    else:
        settings = Settings.from_dict({**PINNED, **fusion, "split": split})
        [spec] = create_jobs([dataset], pipeline, queries, [seed], settings)
        save_job(spec)
        runs[key] = spec.run_id
        save_manifest(manifest)
        print(f"Starting {spec.run_id}", flush=True)
    return spec.run_id if run_job(spec) == "complete" else None


def hybrid_ndcg(run_id: str) -> float:
    metrics = json.loads((paths.RUNS_ROOT / run_id / "metrics.json").read_text())
    return metrics["hybrid"]["ndcg"]["NDCG@10"]


def tune(manifest: dict) -> dict | None:
    """Pick the fusion settings with the best average nDCG@10 across the tuning splits."""
    def average(fusion: dict) -> float | None:
        scores = []
        for dataset, split in TUNING.items():
            run_id = ensure_run(manifest, dataset, "rrf", "all", 42, fusion, split)
            if run_id is None:
                return None
            scores.append(hybrid_ndcg(run_id))
        return mean(scores)

    weight_scores = {}
    for semantic, bm25 in WEIGHTS:
        score = average({**BASELINE, "rrfSemanticWeight": semantic, "rrfBm25Weight": bm25})
        if score is None:
            return None
        weight_scores[(semantic, bm25)] = score
    semantic, bm25 = max(weight_scores, key=weight_scores.get)

    best, best_score = None, None
    for level in ("chunk", "document"):
        for order in ("none", "sum-top2"):
            fusion = {**BASELINE, "rrfSemanticWeight": semantic, "rrfBm25Weight": bm25,
                      "fusionLevel": level, "documentOrder": order}
            score = average(fusion)
            if score is None:
                return None
            if best_score is None or score > best_score:
                best, best_score = fusion, score
    manifest["chosen"] = {"fusion": best, "tuningAverageNdcg10": best_score}
    save_manifest(manifest)
    print(f"Chosen on train/dev: {best} (average nDCG@10 {best_score:.5f})", flush=True)
    return best


def report(manifest: dict, chosen: dict) -> bool:
    variants = {
        "RRF · baseline": ("rrf", BASELINE),
        "RRF · tuned": ("rrf", chosen),
        "RRF + Ettin · baseline": ("rrf-ettin", BASELINE),
        "RRF + Ettin · title": ("rrf-ettin", {**BASELINE, "rerankTitle": "on"}),
        "RRF + Ettin · tuned": ("rrf-ettin", chosen),
        "RRF + Ettin · tuned + title": ("rrf-ettin", {**chosen, "rerankTitle": "on"}),
    }
    table = manifest.setdefault("test", {})
    for dataset, (queries, seed) in TEST.items():
        for label, (pipeline, fusion) in variants.items():
            run_id = ensure_run(manifest, dataset, pipeline, queries, seed, fusion, "test")
            if run_id is None:
                return False
            table.setdefault(dataset, {})[label] = run_id
            save_manifest(manifest)
    return True


def main() -> int:
    manifest = load_manifest()
    chosen = tune(manifest)
    if chosen is None or not report(manifest, chosen):
        print("Stopped; run this script again to resume.", flush=True)
        return 1
    print("All runs complete. Analyze with fusion_experiment/analyze.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
