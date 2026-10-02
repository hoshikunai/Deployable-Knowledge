"""Run the dedup experiment: every dataset x variant on RRF + Ettin, sequentially.

Settings are pinned here rather than taken from the bench GUI, so results don't depend on
whatever was last saved there. Run IDs are recorded in manifest.json after each run; running
the script again skips finished runs and resumes unfinished ones.

    benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/run_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT.parents[1] / "bench-gui"))

import paths  # noqa: E402,F401  (puts the BEIR harness on sys.path)
import corpora  # noqa: E402
from bench import run_job  # noqa: E402
from jobs import Settings, create_jobs, create_prepare_jobs, load_job, save_job  # noqa: E402

MANIFEST = EXPERIMENT_ROOT / "manifest.json"
PIPELINE = "rrf-ettin"
VARIANTS = {
    "A (baseline)": {"dedupMode": "none"},
    "B (exact)": {"dedupMode": "exact"},
    "C (near)": {"dedupMode": "near"},
    "D (≤2 per doc)": {"dedupMode": "document-cap", "maxChunksPerDocument": 2},
}
# ArguAna is sampled: a full Ettin run takes about 70 minutes per variant on this laptop.
DATASETS = {
    "scifact-dup": ("all", 42),
    "scifact": ("all", 42),
    "nfcorpus": ("all", 42),
    "arguana": ("300", 42),
}
PINNED = {
    "retrievalMultiplier": 5, "rerankMultiplier": 2, "rrfRankConstant": 60, "documentDepth": 10,
    "rerankMaxTokens": 512, "chunkMaxTokens": 0, "chunkOverlapTokens": 0,
    "embeddingModel": "nomic-embed-text-v1.5", "timeoutMinutes": 30, "maxSessions": 8,
    # Recorded before titles became the default; keeps re-runs identical to the results.
    "rerankTitle": "off",
}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def prepare_missing_copies() -> bool:
    """Build any dataset's default corpus copy that doesn't exist yet (scifact-dup)."""
    settings = Settings.from_dict(PINNED)
    for dataset in DATASETS:
        status = corpora.variant_status(dataset, None, PINNED["embeddingModel"])
        if status["state"] == "prepared":
            continue
        print(f"Preparing {dataset}: {status['reason']}", flush=True)
        [spec] = create_prepare_jobs([dataset], settings)
        save_job(spec)
        if run_job(spec) != "complete":
            return False
    return True


def main() -> int:
    if not prepare_missing_copies():
        print("A corpus copy could not be prepared; fix the error above and run again.")
        return 1
    manifest = load_manifest()
    for dataset, (queries, seed) in DATASETS.items():
        for variant, overrides in VARIANTS.items():
            run_id = manifest.get(dataset, {}).get(variant)
            if run_id:
                spec = load_job(run_id)
                if spec.state == "complete":
                    continue
                print(f"Resuming {dataset} · {variant}: {run_id}", flush=True)
            else:
                settings = Settings.from_dict({**PINNED, **overrides})
                [spec] = create_jobs([dataset], PIPELINE, queries, [seed], settings)
                save_job(spec)
                manifest.setdefault(dataset, {})[variant] = spec.run_id
                save_manifest(manifest)
                print(f"Starting {dataset} · {variant}: {spec.run_id}", flush=True)
            if run_job(spec) != "complete":
                print(f"Stopped at {dataset} · {variant}; run this script again to resume.")
                return 1
    print("All runs complete. Analyze with dedup_experiment/analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
