"""Prepared BEIR corpora and their chunking variants.

Each variant is a separate app database under .cache/beir/. The default variant
(runtime-<dataset>-001) uses the app's default 1,200-character chunking. Token-chunked
variants are named runtime-<dataset>-t<max>o<overlap> and are built by a prepare job.
The saved BEIR-to-app document mapping lives beside each database so deleting a run
can never remove it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import paths
from dataset_catalog import DATASETS
from run_prepared_benchmarks import PREPARED_MAPPINGS
from run_suite import validate_existing_runtime_schema

DEFAULT_VARIANT = "001"
MAPPING_FILE = "document-id-mapping.json"
MARKER_FILE = "beir-dataset.json"


def variant_id(chunking: dict[str, int] | None) -> str:
    if chunking is None:
        return DEFAULT_VARIANT
    return f"t{chunking['maxTokens']}o{chunking['overlapTokens']}"


def describe_chunking(chunking: dict[str, int] | None) -> str:
    if chunking is None:
        return "1,200-character chunks"
    return f"{chunking['maxTokens']}-token chunks, {chunking['overlapTokens']} overlap"


def runtime_id(dataset: str, chunking: dict[str, int] | None) -> str:
    return f"runtime-{dataset}-{variant_id(chunking)}"


def runtime_path(dataset: str, chunking: dict[str, int] | None) -> Path:
    return paths.RUNTIME_ROOT / runtime_id(dataset, chunking)


def read_marker(runtime: Path) -> dict[str, Any] | None:
    marker = runtime / MARKER_FILE
    if not marker.is_file():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def write_marker(runtime: Path, value: dict[str, Any]) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    temporary = runtime / (MARKER_FILE + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(runtime / MARKER_FILE)


def adopt_legacy_mapping(dataset: str, runtime: Path) -> None:
    """Copy a default corpus's mapping out of its run folder into the runtime, once."""
    target = runtime / MAPPING_FILE
    legacy = PREPARED_MAPPINGS.get(dataset)
    if target.is_file() or legacy is None:
        return
    source = paths.REPOSITORY_ROOT / legacy
    if source.is_file():
        shutil.copy2(source, target)


def mapping_path(dataset: str, chunking: dict[str, int] | None) -> Path:
    runtime = runtime_path(dataset, chunking)
    if chunking is None:
        adopt_legacy_mapping(dataset, runtime)
    return runtime / MAPPING_FILE


def protected_run_folders() -> set[str]:
    """Run folders that still hold a default corpus's original mapping."""
    return {Path(path).parent.name for path in PREPARED_MAPPINGS.values()}


def count_judged_queries(dataset: str, split: str = "test") -> int | None:
    qrels_path = paths.DATASETS_ROOT / dataset / "qrels" / f"{split}.tsv"
    if not qrels_path.is_file():
        return None
    with qrels_path.open(encoding="utf-8", newline="") as stream:
        return len({row["query-id"] for row in csv.DictReader(stream, delimiter="\t")})


def count_chunks(runtime: Path) -> int:
    with sqlite3.connect(f"file:{runtime / 'app.db'}?mode=ro", uri=True) as database:
        return database.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]


def dataset_files_present(dataset: str) -> bool:
    root = paths.DATASETS_ROOT / dataset
    return all((root / name).is_file() for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"))


def variant_status(dataset: str, chunking: dict[str, int] | None) -> dict[str, Any]:
    """Whether this dataset's copy for the chunking is ready, being built, or missing."""
    runtime = runtime_path(dataset, chunking)
    status: dict[str, Any] = {"runtimeId": runtime.name, "state": "missing", "reason": None,
                              "documentCount": None, "chunkCount": None}
    if not dataset_files_present(dataset):
        status["reason"] = "BEIR dataset files are not downloaded"
        return status
    marker = read_marker(runtime)
    if marker is None:
        status["reason"] = "Not prepared for this chunk size"
        return status
    if marker.get("state") == "preparing":
        status["state"] = "preparing"
        status["reason"] = "Prepare job running or interrupted; prepare again to rebuild"
        return status
    mapping = mapping_path(dataset, chunking)
    if not mapping.is_file():
        status["reason"] = f"Saved document mapping is missing: {mapping}"
        return status
    try:
        identity = validate_existing_runtime_schema(runtime, dataset=dataset, mapping=mapping)
    except (RuntimeError, OSError, sqlite3.Error) as error:
        status["reason"] = str(error)
        return status
    status.update(state="prepared", documentCount=identity["documentCount"],
                  chunkCount=marker.get("chunkCount"))
    return status


def list_corpora(chunking: dict[str, int] | None) -> list[dict[str, Any]]:
    corpora = []
    for info in DATASETS:
        status = variant_status(info.name, chunking)
        corpora.append({
            "name": info.name,
            "description": info.description,
            "corpusSize": info.corpus_size,
            "queryCount": count_judged_queries(info.name),
            "prepared": status["state"] == "prepared",
            "chunking": describe_chunking(chunking),
            **status,
        })
    return corpora


def require_prepared(dataset: str, chunking: dict[str, int] | None) -> Path:
    """Return the mapping path for a prepared corpus variant, or raise."""
    status = variant_status(dataset, chunking)
    if status["state"] != "prepared":
        raise ValueError(f"{dataset} ({describe_chunking(chunking)}) is not prepared: {status['reason']}")
    return mapping_path(dataset, chunking)


def start_preparing(dataset: str, chunking: dict[str, int]) -> Path:
    """Create an empty variant runtime, replacing only an unfinished earlier attempt."""
    if not dataset_files_present(dataset):
        raise ValueError(f"{dataset} dataset files are not downloaded")
    runtime = runtime_path(dataset, chunking)
    if runtime.name.endswith(f"-{DEFAULT_VARIANT}"):
        raise ValueError("The default corpus copy is never rebuilt by a prepare job")
    marker = read_marker(runtime)
    if runtime.exists():
        if marker is None or marker.get("dataset") != dataset:
            raise ValueError(f"{runtime} exists but is not a bench corpus copy; move it aside first")
        if marker.get("state") != "preparing":
            raise ValueError(f"{dataset} ({describe_chunking(chunking)}) is already prepared")
        shutil.rmtree(runtime)
    write_marker(runtime, {"dataset": dataset, "chunking": chunking, "state": "preparing"})
    return runtime


def finish_preparing(dataset: str, chunking: dict[str, int], built_mapping: Path) -> dict[str, Any]:
    runtime = runtime_path(dataset, chunking)
    target = runtime / MAPPING_FILE
    shutil.copy2(built_mapping, target)
    identity = validate_existing_runtime_schema(runtime, dataset=dataset, mapping=target)
    marker = {
        "dataset": dataset,
        "chunking": chunking,
        "state": "prepared",
        "documentCount": identity["documentCount"],
        "chunkCount": count_chunks(runtime),
        "mappingSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    write_marker(runtime, marker)
    return marker
