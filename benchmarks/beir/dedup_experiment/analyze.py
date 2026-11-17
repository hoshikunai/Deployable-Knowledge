"""Summarize the dedup experiment from manifest.json into results.md and results.json.

Per dataset x variant x k (5, 10):
- nDCG@k, Recall@k, MRR@k as recorded by BEIR (MRR corrected for ArguAna, see scores.py).
- Distinct-relevant recall@k: relevant duplicate groups found / relevant groups, so the same
  answer returned twice counts once (groups come from duplicate-groups.json for scifact-dup;
  elsewhere each document is its own group).
- Document duplicate rate@k: share of the top-k documents whose group already appeared.
- Chunk repeat rates in the top-k hybrid chunks: same document, exact text, near-duplicate
  (>= 0.9 Jaccard over 5-word shingles), each counted against earlier chunks.
- Mean seconds per query (all three methods) and a paired test of hybrid nDCG@10 against A.

    benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/analyze.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from statistics import mean
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT.parents[1] / "bench-gui"))

import paths  # noqa: E402
from scores import RunData, compare_runs, correct_recorded_mrr, p_value_text  # noqa: E402

K_VALUES = (5, 10)
NEAR_JACCARD = 0.9
SHINGLE_WORDS = 5
BASELINE = "A (baseline)"


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def shingles(text: str) -> set[str]:
    words = normalized(text).split(" ")
    if len(words) <= SHINGLE_WORDS:
        return {" ".join(words)}
    return {" ".join(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def duplicate_groups(dataset: str) -> dict[str, str]:
    path = paths.DATASETS_ROOT / dataset / "duplicate-groups.json"
    if not path.is_file():
        return {}
    return {copy: info["original"] for copy, info in json.loads(path.read_text())["copies"].items()}


def chunk_contents(run: RunData, chunk_ids: set[str]) -> dict[str, str]:
    runtime = (run.config.get("corpus") or {}).get("runtimeId", f"runtime-{run.dataset}-001")
    database = paths.RUNTIME_ROOT / runtime / "app.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        ids = list(chunk_ids)
        contents: dict[str, str] = {}
        for start in range(0, len(ids), 900):
            batch = ids[start:start + 900]
            query = f"SELECT id, content FROM document_chunks WHERE id IN ({','.join('?' * len(batch))})"
            contents.update(dict(connection.execute(query, batch).fetchall()))
    return contents


def document_measures(run: RunData, groups: dict[str, str], k: int) -> dict[str, float]:
    group = lambda doc: groups.get(doc, doc)  # noqa: E731
    duplicate_rates, distinct_recalls = [], []
    for query_id in run.query_ids:
        ranking = run.rankings["hybrid"].get(query_id, {})
        top = [doc for doc, _ in sorted(ranking.items(), key=lambda item: -item[1])
               if doc != query_id][:k]
        seen: set[str] = set()
        repeats = 0
        for doc in top:
            repeats += group(doc) in seen
            seen.add(group(doc))
        duplicate_rates.append(repeats / len(top) if top else 0.0)
        relevant = {group(doc) for doc, grade in run.qrels.get(query_id, {}).items() if grade > 0}
        if relevant:
            distinct_recalls.append(len(relevant & {group(doc) for doc in top}) / len(relevant))
    return {"distinctRelevantRecall": mean(distinct_recalls), "documentDuplicateRate": mean(duplicate_rates)}


def chunk_measures(run: RunData, groups: dict[str, str], k: int) -> dict[str, float]:
    records = run.checkpoint_rankings()
    lists = [record["methods"]["hybrid"][:k] for record in records]
    contents = chunk_contents(run, {hit["chunkId"] for hits in lists for hit in hits})
    same_doc, exact, near, group_repeat, total = 0, 0, 0, 0, 0
    for hits in lists:
        seen_docs, seen_groups, seen_texts, seen_shingles = set(), set(), set(), []
        for hit in hits:
            beir_doc = run.application_to_beir.get(hit["documentId"], hit["documentId"])
            text = contents.get(hit["chunkId"], "")
            text_shingles = shingles(text)
            same_doc += hit["documentId"] in seen_docs
            group_repeat += groups.get(beir_doc, beir_doc) in seen_groups
            exact += normalized(text) in seen_texts
            near += any(jaccard(text_shingles, earlier) >= NEAR_JACCARD for earlier in seen_shingles)
            seen_docs.add(hit["documentId"])
            seen_groups.add(groups.get(beir_doc, beir_doc))
            seen_texts.add(normalized(text))
            seen_shingles.append(text_shingles)
            total += 1
    share = lambda count: count / total if total else 0.0  # noqa: E731
    return {"chunkSameDocument": share(same_doc), "chunkExactText": share(exact),
            "chunkNearDuplicate": share(near), "chunkDuplicateGroup": share(group_repeat)}


def seconds_per_query(run: RunData) -> float:
    timing = json.loads((run.directory / "query-timing.json").read_text())
    return timing["totalSeconds"] / max(1, len(timing["queries"]))


def analyze_run(run_id: str) -> tuple[RunData, dict[str, Any]]:
    run = RunData(run_id)
    correct_recorded_mrr(run)
    groups = duplicate_groups(run.dataset)
    hybrid = run.metrics["hybrid"]
    result: dict[str, Any] = {"runId": run_id, "queries": len(run.query_ids),
                              "secondsPerQuery": seconds_per_query(run), "byK": {}}
    for k in K_VALUES:
        result["byK"][k] = {
            "ndcg": hybrid["ndcg"][f"NDCG@{k}"], "recall": hybrid["recall"][f"Recall@{k}"],
            "mrr": hybrid["mrr"][f"MRR@{k}"],
            **document_measures(run, groups, k), **chunk_measures(run, groups, k),
        }
    return run, result


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    manifest = json.loads((EXPERIMENT_ROOT / "manifest.json").read_text())
    results: dict[str, Any] = {}
    lines = ["# Dedup experiment results", "",
             "Pipeline: RRF + Ettin 32M, 100→40→20, 1,200-character chunks, nomic-embed-text v1.5.",
             "Dedup runs after RRF and before the 40-candidate shortlist cut.", ""]
    for dataset, variants in manifest.items():
        runs = {variant: analyze_run(run_id) for variant, run_id in variants.items()}
        results[dataset] = {variant: data for variant, (_, data) in runs.items()}
        queries = next(iter(runs.values()))[1]["queries"]
        lines += [f"## {dataset} ({queries} queries)", "",
                  "| Variant | k | nDCG | Recall | MRR | Distinct-relevant recall | Doc duplicate rate | "
                  "Chunks: same doc | exact text | near-dup | dup group | s/query |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for variant, (_, data) in runs.items():
            for k in K_VALUES:
                m = data["byK"][k]
                lines.append(
                    f"| {variant} | {k} | {m['ndcg']:.5f} | {m['recall']:.5f} | {m['mrr']:.5f} | "
                    f"{m['distinctRelevantRecall']:.5f} | {percent(m['documentDuplicateRate'])} | "
                    f"{percent(m['chunkSameDocument'])} | {percent(m['chunkExactText'])} | "
                    f"{percent(m['chunkNearDuplicate'])} | {percent(m['chunkDuplicateGroup'])} | "
                    f"{data['secondsPerQuery']:.2f} |")
        baseline = runs.get(BASELINE)
        if baseline:
            lines += ["", "Hybrid nDCG@10 vs A (paired sign-flip test):", ""]
            for variant, (run, _) in runs.items():
                if variant == BASELINE:
                    continue
                comparison = compare_runs(baseline[0], run)
                if comparison:
                    results[dataset][variant]["vsBaseline"] = comparison
                    lines.append(f"- {variant}: Δ {comparison['delta']:+.5f}, p = "
                                 f"{p_value_text(comparison['pValue'])}, wins/ties/losses "
                                 f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}")
        lines.append("")
    (EXPERIMENT_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (EXPERIMENT_ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
