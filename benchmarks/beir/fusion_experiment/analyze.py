"""Summarize the fusion experiment from manifest.json into results.md and results.json.

- Tuning: hybrid nDCG@10 / Recall@10 / MRR@10 for every configuration run on SciFact train and
  NFCorpus dev, with the average used to choose.
- Test: each variant against its pipeline's baseline (paired sign-flip test on nDCG@10), plus
  two diagnostics for the top-10 documents: average chunks per document (a high value means
  long documents are favoured, a risk of sum-top2) and the share found by both retrievers.

    benchmarks/beir/.venv/bin/python benchmarks/beir/fusion_experiment/analyze.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT.parents[1] / "bench-gui"))

import paths  # noqa: E402
from scores import RunData, compare_runs, correct_recorded_mrr, p_value_text  # noqa: E402

BASELINES = {"RRF": "RRF · baseline", "RRF + Ettin": "RRF + Ettin · baseline"}


def load_run(run_id: str) -> RunData:
    run = RunData(run_id)
    correct_recorded_mrr(run)
    return run


def headline(run: RunData) -> dict[str, float]:
    hybrid = run.metrics["hybrid"]
    return {"ndcg10": hybrid["ndcg"]["NDCG@10"], "recall10": hybrid["recall"]["Recall@10"],
            "mrr10": hybrid["mrr"]["MRR@10"]}


def fusion_label(config: dict[str, Any]) -> str:
    pipeline = config.get("hybridPipeline") or {}
    parts = [f"weights {pipeline.get('rrfSemanticWeight', 1):g}:{pipeline.get('rrfBm25Weight', 1):g}",
             f"{pipeline.get('fusionLevel', 'chunk')} fusion",
             f"order {pipeline.get('documentOrder', 'none')}"]
    return " · ".join(parts)


def chunks_per_document(run: RunData) -> dict[str, int]:
    runtime = (run.config.get("corpus") or {}).get("runtimeId", f"runtime-{run.dataset}-001")
    database = paths.RUNTIME_ROOT / runtime / "app.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        counts = connection.execute(
            "SELECT document_id, COUNT(*) FROM document_chunks GROUP BY document_id").fetchall()
    beir_counts: dict[str, int] = {}
    for application_id, count in counts:
        beir_id = run.application_to_beir.get(application_id)
        if beir_id is not None:
            beir_counts[beir_id] = count
    return beir_counts


def top_documents(run: RunData, method: str, query_id: str, k: int = 10) -> list[str]:
    ranking = run.rankings[method].get(query_id, {})
    return [doc for doc, _ in sorted(ranking.items(), key=lambda item: -item[1]) if doc != query_id][:k]


def diagnostics(run: RunData, lengths: dict[str, int]) -> dict[str, float]:
    chunk_counts, both = [], []
    for query_id in run.query_ids:
        top = top_documents(run, "hybrid", query_id)
        if not top:
            continue
        chunk_counts.append(mean(lengths.get(doc, 1) for doc in top))
        found = set(top_documents(run, "bm25", query_id)) & set(top_documents(run, "semantic", query_id))
        both.append(sum(doc in found for doc in top) / len(top))
    return {"chunksPerTop10Document": mean(chunk_counts), "foundByBothRetrievers": mean(both)}


def tuning_section(manifest: dict, lines: list[str]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, run_id in manifest.get("runs", {}).items():
        dataset, pipeline, _queries, split, _ = json.loads(key)
        if split == "test":
            continue
        run = load_run(run_id)
        rows[fusion_label(run.config)][f"{dataset} {split}"] = {**headline(run), "runId": run_id}
    columns = sorted({name for row in rows.values() for name in row})
    lines += ["## Tuning (RRF only)", "",
              "| Configuration | " + " | ".join(f"{c} nDCG@10" for c in columns) + " | Average |",
              "|" + "---|" * (len(columns) + 2)]
    for label, row in sorted(rows.items(), key=lambda item: -mean(v["ndcg10"] for v in item[1].values())):
        values = [row[c]["ndcg10"] for c in columns if c in row]
        lines.append(f"| {label} | " + " | ".join(
            f"{row[c]['ndcg10']:.5f}" if c in row else "–" for c in columns) + f" | {mean(values):.5f} |")
    chosen = manifest.get("chosen")
    if chosen:
        lines += ["", f"Chosen: `{json.dumps(chosen['fusion'])}` "
                      f"(average nDCG@10 {chosen['tuningAverageNdcg10']:.5f})."]
    lines.append("")
    return rows


def test_section(manifest: dict, lines: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dataset, variants in manifest.get("test", {}).items():
        runs = {label: load_run(run_id) for label, run_id in variants.items()}
        lengths = chunks_per_document(next(iter(runs.values())))
        queries = len(next(iter(runs.values())).query_ids)
        lines += [f"## {dataset} test ({queries} queries)", "",
                  "| Variant | nDCG@10 | Recall@10 | MRR@10 | Δ nDCG vs baseline | p | "
                  "Wins/ties/losses | Chunks per top-10 doc | Found by both |",
                  "|---|---|---|---|---|---|---|---|---|"]
        results[dataset] = {}
        for label, run in runs.items():
            data = {**headline(run), **diagnostics(run, lengths), "runId": run.id}
            baseline_label = BASELINES["RRF + Ettin" if label.startswith("RRF + Ettin") else "RRF"]
            comparison = None
            if label != baseline_label and baseline_label in runs:
                comparison = compare_runs(runs[baseline_label], run)
                data["vsBaseline"] = comparison
            results[dataset][label] = data
            delta = (f"{comparison['delta']:+.5f} | {p_value_text(comparison['pValue'])} | "
                     f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
                     if comparison else "– | – | –")
            lines.append(f"| {label} | {data['ndcg10']:.5f} | {data['recall10']:.5f} | "
                         f"{data['mrr10']:.5f} | {delta} | {data['chunksPerTop10Document']:.2f} | "
                         f"{data['foundByBothRetrievers'] * 100:.1f}% |")
        lines.append("")
    return results


def main() -> None:
    manifest = json.loads((EXPERIMENT_ROOT / "manifest.json").read_text())
    lines = ["# Fusion experiment results", "",
             "RRF weights, fusion level, document ordering and title input. 100→40→20, "
             "1,200-character chunks, nomic-embed-text v1.5. Tuned on SciFact train and NFCorpus "
             "dev; reported on test.", ""]
    tuning = tuning_section(manifest, lines)
    test = test_section(manifest, lines)
    (EXPERIMENT_ROOT / "results.json").write_text(
        json.dumps({"tuning": tuning, "test": test}, indent=2) + "\n")
    (EXPERIMENT_ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
