"""List, describe, and delete benchmark run directories."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import paths
from corpora import describe_chunking, protected_run_folders
from jobs import DEFAULT_RERANK_MAX_TOKENS, PIPELINES, job_path

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*-[A-Za-z0-9][A-Za-z0-9._-]*$")


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(run_id: str) -> Path:
    if not RUN_ID_PATTERN.match(run_id) or ".." in run_id:
        raise ValueError(f"Invalid run id: {run_id!r}")
    directory = paths.RUNS_ROOT / run_id
    if not directory.is_dir():
        raise FileNotFoundError(f"No run named {run_id}")
    return directory


def pipeline_key(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None
    for key, pipeline in PIPELINES.items():
        if pipeline["reranker"] == config["reranker"]:
            return key
    return None


def describe_pipeline(config: dict[str, Any] | None, http_top_k: int | None) -> str:
    """Human label such as 'RRF + Ettin 32M · 100→40→20'."""
    if not config:
        return "Unknown pipeline (run predates pipeline recording)"
    key = pipeline_key(config)
    label = PIPELINES[key]["label"] if key else f"RRF + {config['reranker']}"
    if http_top_k is None:
        return label
    depths = [str(http_top_k * config["retrievalMultiplier"])]
    if config["reranker"]:
        depths.append(str(http_top_k * config["rerankMultiplier"]))
    depths.append(str(http_top_k))
    suffix = "" if config["rrfRankConstant"] == 60 else f" · k={config['rrfRankConstant']}"
    rerank_tokens = config.get("rerankMaxTokens", DEFAULT_RERANK_MAX_TOKENS)
    if config["reranker"] and rerank_tokens != DEFAULT_RERANK_MAX_TOKENS:
        suffix += f" · reranker reads {rerank_tokens} tokens"
    return f"{label} · {'→'.join(depths)}{suffix}"


def describe_corpus(config: dict[str, Any]) -> str:
    """Runs before corpus recording all searched the default character-chunked copies."""
    corpus = config.get("corpus")
    return describe_chunking(corpus["chunking"] if corpus else None)


def has_chunk_scores(directory: Path) -> bool:
    checkpoint_path = directory / "query-checkpoints.jsonl"
    if not checkpoint_path.is_file():
        return False
    with checkpoint_path.open(encoding="utf-8") as stream:
        first = stream.readline()
    if not first.strip():
        return False
    raw = json.loads(first).get("rawRanking") or {}
    hits = raw.get("methods", {}).get("hybrid", [])
    return bool(hits) and hits[0].get("score") is not None


def summarize_run(directory: Path) -> dict[str, Any]:
    config = read_json(directory / "run-config.json") or {}
    status = read_json(directory / "run-status.json") or {}
    metrics = read_json(directory / "metrics.json")
    job = read_json(job_path(directory.name))
    pipeline = config.get("hybridPipeline")
    query_ids = config.get("queryIds") or []
    ndcg = {
        method: metrics[method]["ndcg"].get("NDCG@10")
        for method in ("bm25", "semantic", "hybrid")
        if metrics and method in metrics
    }
    return {
        "id": directory.name,
        "dataset": config.get("dataset") or directory.name.split("-", 1)[0],
        "pipeline": pipeline_key(pipeline),
        "pipelineConfig": pipeline,
        "pipelineLabel": describe_pipeline(pipeline, config.get("httpTopK")),
        "corpusLabel": describe_corpus(config),
        "queryCount": len(query_ids),
        "selection": config.get("selectionProtocol"),
        "sampleSeed": config.get("sampleSeed"),
        "status": status.get("status", "unknown"),
        "completed": len(status.get("completedQueryIds", [])),
        "ndcg10": ndcg,
        "hasScores": has_chunk_scores(directory),
        "modified": directory.stat().st_mtime,
        "jobState": job.get("state") if job else None,
        "jobMessage": job.get("message") if job else None,
        "resumable": job is not None,
    }


def list_runs() -> list[dict[str, Any]]:
    if not paths.RUNS_ROOT.is_dir():
        return []
    directories = [
        directory for directory in paths.RUNS_ROOT.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
        and (directory / "run-config.json").is_file()
        # Ingestion folders of corpus copies are not benchmark runs
        and (read_json(directory / "run-status.json") or {}).get("status") != "prepared"
    ]
    return sorted((summarize_run(directory) for directory in directories),
                  key=lambda run: run["modified"], reverse=True)


def delete_run(run_id: str) -> list[str]:
    directory = run_path(run_id)
    if run_id in protected_run_folders():
        raise ValueError(f"{run_id} holds a prepared corpus's original document mapping; "
                         "it is kept so the corpus stays usable")
    removed = [str(directory.relative_to(paths.REPOSITORY_ROOT))]
    shutil.rmtree(directory)
    supervisor = paths.RUNS_ROOT / f".supervisor-{run_id}"
    if supervisor.is_dir():
        shutil.rmtree(supervisor)
        removed.append(str(supervisor.relative_to(paths.REPOSITORY_ROOT)))
    job = job_path(run_id)
    if job.is_file():
        job.unlink()
        removed.append(str(job.relative_to(paths.REPOSITORY_ROOT)))
    return removed
