"""Scores for one or more runs: metrics, paired comparisons, and chunk-score calibration.

Every result is returned twice: as raw JSON and as a Markdown document of tables.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytrec_eval

import paths
from run import canonicalize_qrels, ranking_unit
from corpora import DEFAULT_DOCUMENT_PARSING, DEFAULT_EMBEDDING_MODEL
from runs import describe_corpus, describe_pipeline, read_json, run_path

METHODS = ("bm25", "semantic", "hybrid")
METRIC_COLUMNS = (
    ("nDCG", "ndcg", "NDCG@10"),
    ("Recall", "recall", "Recall@10"),
    ("MRR", "mrr", "MRR@10"),
    ("MAP", "map", "MAP@10"),
    ("P", "precision", "P@10"),
)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PERMUTATIONS = 20000


def load_qrels(dataset: str, split: str = "test") -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with (paths.DATASETS_ROOT / dataset / "qrels" / f"{split}.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


class RunData:
    """Artifacts of one run, loaded once."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.directory: Path = run_path(run_id)
        self.config: dict[str, Any] = read_json(self.directory / "run-config.json") or {}
        self.status: dict[str, Any] = read_json(self.directory / "run-status.json") or {}
        self.metrics: dict[str, Any] | None = read_json(self.directory / "metrics.json")
        self.rankings: dict[str, Any] | None = read_json(self.directory / "document-rankings.json")
        self.dataset: str = self.config["dataset"]
        self.pipeline = self.config.get("hybridPipeline")
        self.label = (f"{describe_pipeline(self.pipeline, self.config.get('httpTopK'))} · "
                      f"{describe_corpus(self.config)}")
        mapping = read_json(self.directory / "document-id-mapping.json") or {}
        self.application_to_beir: dict[str, str] = mapping.get("applicationToBeir", {})
        self.page_units = self.config.get("rankingUnit") == "page"
        qrels = load_qrels(self.dataset, self.config.get("split", "test"))
        self.qrels = qrels if self.page_units else canonicalize_qrels(
            qrels,
            mapping.get("beirToApplication", {}),
            self.application_to_beir,
            set(mapping.get("missingCorpusQrelDocumentIds", [])),
        )

    @property
    def query_ids(self) -> list[str]:
        return self.config.get("queryIds", [])

    def checkpoint_rankings(self) -> list[dict[str, Any]]:
        path = self.directory / "query-checkpoints.jsonl"
        if not path.is_file():
            return []
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if record.get("failure") is None and record.get("rawRanking"):
                        records.append(record["rawRanking"])
        return records


def per_query_ndcg(run: RunData, method: str) -> dict[str, float]:
    """nDCG@10 per query, matching BEIR's evaluator (identical query/document IDs removed)."""
    if run.rankings is None:
        return {}
    results = {
        query_id: {doc: score for doc, score in ranking.items() if doc != query_id}
        for query_id, ranking in run.rankings[method].items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(run.qrels, {"ndcg_cut.10"})
    return {query_id: values["ndcg_cut_10"] for query_id, values in evaluator.evaluate(results).items()}


def recompute_mrr(run: RunData, method: str) -> dict[str, float]:
    """BEIR's MRR@k from saved rankings, without each query's own document.

    Runs recorded before the harness fix counted that document (ArguAna queries are also
    corpus documents) as a miss at rank 1; nDCG, MAP, Recall, and P always excluded it.
    """
    k_values = run.config.get("kValues") or [1, 3, 5, 10]
    rankings = (run.rankings or {}).get(method, {})
    totals = {k: 0.0 for k in k_values}
    for query_id in run.query_ids:
        ranking = sorted(((doc, score) for doc, score in rankings.get(query_id, {}).items()
                          if doc != query_id), key=lambda item: item[1], reverse=True)
        relevant = {doc for doc, grade in run.qrels.get(query_id, {}).items() if grade > 0}
        for k in k_values:
            rank = next((index for index, (doc, _) in enumerate(ranking[:k]) if doc in relevant), None)
            if rank is not None:
                totals[k] += 1.0 / (rank + 1)
    count = len(run.query_ids) or 1
    return {f"MRR@{k}": round(totals[k] / count, 5) for k in k_values}


def correct_recorded_mrr(run: RunData) -> dict[str, dict[str, Any]]:
    """Replace MRR recorded before the fix with the recomputed value; return what changed."""
    if not run.metrics or run.rankings is None:
        return {}
    corrections: dict[str, dict[str, Any]] = {}
    corrected_metrics = copy.deepcopy(run.metrics)
    for method in METHODS:
        if method not in corrected_metrics:
            continue
        recorded = corrected_metrics[method]["mrr"]
        recomputed = recompute_mrr(run, method)
        if any(abs(recorded.get(key, 0.0) - value) > 1e-5 for key, value in recomputed.items()):
            corrections[method] = {"recorded": recorded, "corrected": recomputed}
            corrected_metrics[method]["mrr"] = recomputed
    run.metrics = corrected_metrics
    return corrections


def sign_flip_p_value(differences: np.ndarray) -> float:
    if differences.size == 0 or not differences.any():
        return 1.0
    generator = np.random.default_rng(0)
    signs = generator.choice([-1.0, 1.0], size=(PERMUTATIONS, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    return float((null >= abs(differences.mean()) - 1e-12).mean())


def corpus_differences(baseline: RunData, other: RunData) -> list[str]:
    """What differs between the two runs' corpus copies, which changes retriever results."""
    def corpus(run: RunData) -> dict[str, Any]:
        return run.config.get("corpus") or {}

    differences = []
    if corpus(baseline).get("chunking") != corpus(other).get("chunking"):
        differences.append("chunk size")
    base_model = corpus(baseline).get("embeddingModel", DEFAULT_EMBEDDING_MODEL)
    if base_model != corpus(other).get("embeddingModel", DEFAULT_EMBEDDING_MODEL):
        differences.append("embedding model")
    base_parsing = corpus(baseline).get("documentParsing", DEFAULT_DOCUMENT_PARSING)
    if base_parsing != corpus(other).get("documentParsing", DEFAULT_DOCUMENT_PARSING):
        differences.append("document parsing")
    return differences


def retriever_check(bm25_same: bool, semantic_same: bool, differences: list[str]) -> tuple[str, bool]:
    """Describe whether BM25 and semantic results match, and whether any mismatch is expected.

    Chunking and document parsing change both retrievers; the embedding model changes only
    semantic search.
    """
    changes_text = {"chunk size", "document parsing"} & set(differences)
    unexpected = (not bm25_same and not changes_text) or (
        not semantic_same and not differences)
    if unexpected:
        return "NO — retrievers differ unexpectedly; check the corpus copies", True
    if bm25_same and semantic_same:
        return "yes", False
    changed = [name for name, same in (("BM25", bm25_same), ("semantic", semantic_same)) if not same]
    verb = "differs" if len(changed) == 1 else "differ"
    return f"{' and '.join(changed)} {verb} by design ({', '.join(differences)})", False


def compare_runs(baseline: RunData, other: RunData) -> dict[str, Any] | None:
    if baseline.dataset != other.dataset or baseline.query_ids != other.query_ids:
        return None
    if baseline.rankings is None or other.rankings is None:
        return None
    base = per_query_ndcg(baseline, "hybrid")
    candidate = per_query_ndcg(other, "hybrid")
    query_ids = sorted(set(base) & set(candidate))
    a = np.array([base[query_id] for query_id in query_ids])
    b = np.array([candidate[query_id] for query_id in query_ids])
    differences = b - a
    bm25_same = baseline.rankings["bm25"] == other.rankings["bm25"]
    semantic_same = baseline.rankings["semantic"] == other.rankings["semantic"]
    corpus_changes = corpus_differences(baseline, other)
    retrievers, unexpected = retriever_check(bm25_same, semantic_same, corpus_changes)
    return {
        "baseline": baseline.id,
        "candidate": other.id,
        "queries": len(query_ids),
        "baselineNdcg10": float(a.mean()),
        "candidateNdcg10": float(b.mean()),
        "delta": float(differences.mean()),
        "pValue": sign_flip_p_value(differences),
        "wins": int((differences > 1e-9).sum()),
        "ties": int((np.abs(differences) <= 1e-9).sum()),
        "losses": int((differences < -1e-9).sum()),
        "bm25Identical": bm25_same,
        "semanticIdentical": semantic_same,
        "corpusDifferences": corpus_changes,
        "retrievers": retrievers,
        "unexpectedRetrieverDifference": unexpected,
    }


def labeled_chunks(run: RunData, record: dict[str, Any], method: str) -> list[tuple[float, bool]] | None:
    query_id = record["queryId"]
    judgments = run.qrels.get(query_id, {})
    chunks = []
    for hit in record["methods"].get(method, []):
        if hit.get("score") is None:
            return None
        mapped = hit["documentId"] in run.application_to_beir
        document = ranking_unit(hit, run.application_to_beir, run.page_units) if mapped else None
        if document == query_id:
            continue
        chunks.append((float(hit["score"]), judgments.get(document, 0) > 0))
    if method == "bm25" and chunks:
        top = max(score for score, _ in chunks) or 1.0
        chunks = [(score / top, relevant) for score, relevant in chunks]
    return chunks


def calibrate(run: RunData) -> dict[str, Any] | None:
    records = run.checkpoint_rankings()
    if not records:
        return None
    calibration: dict[str, Any] = {"queries": len(records), "methods": {}}
    for method in METHODS:
        per_query = [labeled_chunks(run, record, method) for record in records]
        if any(chunks is None for chunks in per_query):
            return None
        rows = []
        for threshold in (0.0, *THRESHOLDS):
            kept = [[relevant for score, relevant in chunks if score >= threshold]
                    for chunks in per_query]
            kept_total = sum(len(query) for query in kept)
            relevant_total = sum(sum(query) for query in kept)
            rows.append({
                "threshold": threshold,
                "avgChunksKept": kept_total / len(kept),
                "keptRelevantShare": relevant_total / kept_total if kept_total else None,
                "queriesWithRelevant": sum(any(query) for query in kept) / len(kept),
                "queriesEmpty": sum(not query for query in kept) / len(kept),
            })
        relevant_scores = [score for chunks in per_query for score, relevant in chunks if relevant]
        other_scores = [score for chunks in per_query for score, relevant in chunks if not relevant]
        calibration["methods"][method] = {
            "scoreScale": "score ÷ top score in query" if method == "bm25" else "raw score",
            "meanRelevantScore": float(np.mean(relevant_scores)) if relevant_scores else None,
            "meanNonRelevantScore": float(np.mean(other_scores)) if other_scores else None,
            "rows": rows,
        }
    return calibration


def percent(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def number(value: float | None, digits: int = 5) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def run_markdown(run: RunData, calibration: dict[str, Any] | None,
                 mrr_corrections: dict[str, dict[str, Any]]) -> list[str]:
    total = len(run.query_ids)
    completed = len(run.status.get("completedQueryIds", []))
    selection = "all" if run.config.get("selectionProtocol") != "seeded-shuffle" else (
        f"seed {run.config.get('sampleSeed')}")
    selection += f", {run.config.get('split', 'test')} split"
    lines = [f"## {run.dataset} · {run.label}",
             f"`{run.id}` · {run.status.get('status', 'unknown')} · "
             f"{completed}/{total} queries ({selection})", ""]
    if run.metrics:
        lines += ["### Retrieval metrics @10", ""]
        lines += markdown_table(
            ["Method", *(name for name, _, _ in METRIC_COLUMNS)],
            [[method, *(number(run.metrics[method][group].get(key)) for _, group, key in METRIC_COLUMNS)]
             for method in METHODS if method in run.metrics],
        )
        lines.append("")
        if mrr_corrections:
            recorded = ", ".join(f"{method} {values['recorded'].get('MRR@10', 0):.5f}"
                                 for method, values in mrr_corrections.items())
            lines += ["_MRR recomputed from saved rankings: this run was recorded before the "
                      "harness stopped counting each query's own document as a miss. "
                      f"Recorded MRR@10: {recorded}._", ""]
    else:
        lines += ["_No metrics yet: the run has not finished._", ""]
    if calibration is None:
        lines += ["_No chunk scores recorded (run predates score recording)._", ""]
        return lines
    for method in ("hybrid", "semantic", "bm25"):
        data = calibration["methods"][method]
        lines += [f"### Chunk-score calibration: {method} ({data['scoreScale']}, "
                  f"{calibration['queries']} queries)", ""]
        lines += markdown_table(
            ["Threshold", "Chunks kept / query", "Kept that are relevant",
             "Queries with a relevant chunk", "Queries left empty"],
            [["none" if row["threshold"] == 0 else f"≥ {row['threshold']:.1f}",
              f"{row['avgChunksKept']:.1f}", percent(row["keptRelevantShare"]),
              percent(row["queriesWithRelevant"]), percent(row["queriesEmpty"])]
             for row in data["rows"]],
        )
        lines += ["", f"Mean score — relevant: {number(data['meanRelevantScore'], 3)} · "
                      f"not relevant: {number(data['meanNonRelevantScore'], 3)}", ""]
    return lines


def p_value_text(value: float) -> str:
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def comparison_markdown(comparisons: list[dict[str, Any]]) -> list[str]:
    if not comparisons:
        return []
    lines = ["## Paired comparison (hybrid nDCG@10)", ""]
    lines += markdown_table(
        ["Baseline", "Candidate", "Queries", "Baseline nDCG", "Candidate nDCG", "Δ", "p",
         "Wins / ties / losses", "BM25 & semantic results"],
        [[f"`{item['baseline']}`", f"`{item['candidate']}`", str(item["queries"]),
          number(item["baselineNdcg10"]), number(item["candidateNdcg10"]),
          f"{item['delta']:+.5f}", p_value_text(item["pValue"]),
          f"{item['wins']} / {item['ties']} / {item['losses']}",
          item["retrievers"]]
         for item in comparisons],
    )
    lines += ["", "p is a paired sign-flip test on per-query nDCG@10; below 0.05 is unlikely to be chance.", ""]
    return lines


def build_scores(run_ids: list[str]) -> dict[str, Any]:
    if not run_ids:
        raise ValueError("Select at least one run")
    runs = [RunData(run_id) for run_id in run_ids]
    raw_runs = []
    markdown: list[str] = []
    for run in runs:
        mrr_corrections = correct_recorded_mrr(run)
        calibration = calibrate(run)
        raw_runs.append({
            "id": run.id,
            "dataset": run.dataset,
            "pipeline": run.label,
            "pipelineConfig": run.pipeline,
            "status": run.status.get("status"),
            "queryCount": len(run.query_ids),
            "sampleSeed": run.config.get("sampleSeed"),
            "metrics": run.metrics,
            "mrrCorrections": mrr_corrections,
            "calibration": calibration,
        })
        markdown += run_markdown(run, calibration, mrr_corrections)
    comparisons = [
        comparison
        for other in runs[1:]
        if (comparison := compare_runs(runs[0], other)) is not None
    ]
    markdown += comparison_markdown(comparisons)
    skipped = [other.id for other in runs[1:]
               if not any(item["candidate"] == other.id for item in comparisons)]
    if len(runs) > 1 and skipped:
        markdown += [f"_Not compared with the first run (different dataset, query sample, or "
                     f"unfinished): {', '.join(skipped)}_", ""]
    return {"raw": {"runs": raw_runs, "comparisons": comparisons},
            "markdown": "\n".join(markdown).strip() + "\n"}
