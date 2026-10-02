"""Prepared BEIR corpora and their variants (chunking and embedding model).

Each variant is a separate app database under .cache/beir/. The default variant
(runtime-<dataset>-001) uses the app's defaults: 1,200-character chunks embedded with
nomic-embed-text v1.5. Other variants are built by a prepare job and named from what
differs, e.g. runtime-<dataset>-t256o32, runtime-<dataset>-bekko-a25m, or both.
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

# IDs mirror src/lib/server/rag/embedding-models/registry.ts; the app rejects unknown IDs.
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "nomic-embed-text-v1.5": {"label": "nomic-embed-text v1.5 (768-d)", "tag": None,
                              "maxTokens": 8192},
    "bekko-embedding-v1-a25m": {"label": "bekko-embedding v1 a25m (384-d)", "tag": "bekko-a25m",
                                "maxTokens": 8192},
    "bge-base-en-v1.5": {"label": "bge-base-en v1.5 (768-d)", "tag": "bge-base",
                         "maxTokens": 512},
    "bge-small-en-v1.5": {"label": "bge-small-en v1.5 (384-d)", "tag": "bge-small",
                          "maxTokens": 512},
    "granite-embedding-97m-multilingual-r2": {
        "label": "granite-embedding 97m multilingual r2 (384-d)", "tag": "granite-97m",
        "maxTokens": 32768},
    "granite-embedding-english-r2": {
        "label": "granite-embedding english r2 (768-d)", "tag": "granite-en", "maxTokens": 8192},
    "jina-embeddings-v5-text-nano-retrieval": {
        "label": "jina-embeddings v5 text nano retrieval (768-d, non-commercial)",
        "tag": "jina-v5-nano", "maxTokens": 8192},
}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1.5"
Chunking = dict[str, int] | None
MAPPING_FILE = "document-id-mapping.json"
MARKER_FILE = "beir-dataset.json"


def variant_id(chunking: Chunking, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    parts = []
    if chunking is not None:
        parts.append(f"t{chunking['maxTokens']}o{chunking['overlapTokens']}")
    tag = EMBEDDING_MODELS[embedding_model]["tag"]
    if tag is not None:
        parts.append(tag)
    return "-".join(parts) or DEFAULT_VARIANT


def describe_chunking(chunking: Chunking) -> str:
    if chunking is None:
        return "1,200-character chunks"
    return f"{chunking['maxTokens']}-token chunks, {chunking['overlapTokens']} overlap"


def describe_variant(chunking: Chunking, embedding_model: str) -> str:
    label = EMBEDDING_MODELS.get(embedding_model, {}).get("label", embedding_model)
    return f"{describe_chunking(chunking)} · {label}"


def runtime_id(dataset: str, chunking: Chunking,
               embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    return f"runtime-{dataset}-{variant_id(chunking, embedding_model)}"


def runtime_path(dataset: str, chunking: Chunking,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    return paths.RUNTIME_ROOT / runtime_id(dataset, chunking, embedding_model)


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


def mapping_path(dataset: str, chunking: Chunking,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    runtime = runtime_path(dataset, chunking, embedding_model)
    if runtime.name.endswith(f"-{DEFAULT_VARIANT}"):
        adopt_legacy_mapping(dataset, runtime)
    return runtime / MAPPING_FILE


def protected_run_folders() -> set[str]:
    """Run folders that still hold a default corpus's original mapping."""
    return {Path(path).parent.name for path in PREPARED_MAPPINGS.values()}


def has_split(dataset: str, split: str) -> bool:
    return (paths.DATASETS_ROOT / dataset / "qrels" / f"{split}.tsv").is_file()


def available_splits(dataset: str) -> list[str]:
    return [split for split in ("test", "train", "dev") if has_split(dataset, split)]


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


def variant_status(dataset: str, chunking: Chunking,
                   embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, Any]:
    """Whether this dataset's copy for the variant is ready, being built, or missing."""
    runtime = runtime_path(dataset, chunking, embedding_model)
    status: dict[str, Any] = {"runtimeId": runtime.name, "state": "missing", "reason": None,
                              "documentCount": None, "chunkCount": None}
    if not dataset_files_present(dataset):
        status["reason"] = "BEIR dataset files are not downloaded"
        return status
    marker = read_marker(runtime)
    if marker is None:
        status["reason"] = "Not prepared for this chunk size and embedding model"
        return status
    marker_model = marker.get("embeddingModel", DEFAULT_EMBEDDING_MODEL)
    if marker_model != embedding_model:
        status["reason"] = f"Copy was embedded with {marker_model}, not {embedding_model}"
        return status
    if marker.get("state") == "preparing":
        status["state"] = "preparing"
        status["reason"] = "Prepare job running or interrupted; prepare again to rebuild"
        return status
    mapping = mapping_path(dataset, chunking, embedding_model)
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


def list_corpora(chunking: Chunking, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 split: str = "test") -> list[dict[str, Any]]:
    corpora = []
    for info in DATASETS:
        status = variant_status(info.name, chunking, embedding_model)
        corpora.append({
            "name": info.name,
            "description": info.description,
            "corpusSize": info.corpus_size,
            "queryCount": count_judged_queries(info.name, split),
            "splits": available_splits(info.name),
            "prepared": status["state"] == "prepared",
            "chunking": describe_chunking(chunking),
            "variant": describe_variant(chunking, embedding_model),
            **status,
        })
    return corpora


def require_prepared(dataset: str, chunking: Chunking,
                     embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    """Return the mapping path for a prepared corpus variant, or raise."""
    status = variant_status(dataset, chunking, embedding_model)
    if status["state"] != "prepared":
        variant = describe_variant(chunking, embedding_model)
        raise ValueError(f"{dataset} ({variant}) is not prepared: {status['reason']}")
    return mapping_path(dataset, chunking, embedding_model)


def start_preparing(dataset: str, chunking: Chunking,
                    embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Path:
    """Create an empty variant runtime, replacing only an unfinished earlier attempt."""
    if not dataset_files_present(dataset):
        raise ValueError(f"{dataset} dataset files are not downloaded")
    runtime = runtime_path(dataset, chunking, embedding_model)
    marker = read_marker(runtime)
    # A default copy can be built for a corpus that has none (or retried after an unfinished
    # attempt), but a finished one is never rebuilt: earlier benchmark results depend on it.
    unfinished = marker is not None and marker.get("state") == "preparing"
    if runtime.name.endswith(f"-{DEFAULT_VARIANT}") and runtime.exists() and not unfinished:
        raise ValueError(f"{runtime.name} already exists; the default copy is never rebuilt")
    if runtime.exists():
        if marker is None or marker.get("dataset") != dataset:
            raise ValueError(f"{runtime} exists but is not a bench corpus copy; move it aside first")
        if marker.get("state") != "preparing":
            variant = describe_variant(chunking, embedding_model)
            raise ValueError(f"{dataset} ({variant}) is already prepared")
        shutil.rmtree(runtime)
    write_marker(runtime, {"dataset": dataset, "chunking": chunking,
                           "embeddingModel": embedding_model, "state": "preparing"})
    return runtime


def finish_preparing(dataset: str, chunking: Chunking, embedding_model: str,
                     built_mapping: Path) -> dict[str, Any]:
    runtime = runtime_path(dataset, chunking, embedding_model)
    target = runtime / MAPPING_FILE
    shutil.copy2(built_mapping, target)
    identity = validate_existing_runtime_schema(runtime, dataset=dataset, mapping=target)
    marker = {
        "dataset": dataset,
        "chunking": chunking,
        "embeddingModel": embedding_model,
        "state": "prepared",
        "documentCount": identity["documentCount"],
        "chunkCount": count_chunks(runtime),
        "mappingSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    write_marker(runtime, marker)
    return marker
