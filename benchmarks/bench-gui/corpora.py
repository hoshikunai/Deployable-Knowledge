"""Prepared BEIR corpora and their variants (chunking, embedding model, document parsing).

Each variant is a separate app database under .cache/beir/. The default variant
(runtime-<dataset>-001) uses the app's defaults: 1,200-character chunks embedded with
nomic-embed-text v1.5, PDFs parsed without table recognition. Other variants are built by a
prepare job and named from what differs, e.g. runtime-<dataset>-t256o32,
runtime-<dataset>-bge-small, runtime-pdf-eval-docling, or several of these.
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
from dataset_catalog import DATASETS, DOCUMENT_PARSING_MODES, is_pdf_dataset
from run_prepared_benchmarks import PREPARED_MAPPINGS
from run_suite import validate_existing_runtime_schema

DEFAULT_VARIANT = "001"

# nomic is the app's llama.cpp GGUF default (its token limit is the GGUF context minus 3);
# other IDs mirror BENCHMARK_EMBEDDING_MODELS in src/lib/server/embeddings/transformers.ts.
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "nomic-embed-text-v1.5": {"label": "nomic-embed-text v1.5 (768-d)", "tag": None,
                              "maxTokens": 2045},
    "bge-small-en-v1.5": {"label": "bge-small-en v1.5 (384-d)", "tag": "bge-small",
                          "maxTokens": 512},
}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1.5"
# Ingestion-time PDF parsing (the app's table recognition switch). Only PDF datasets have
# documents to parse, so a non-default mode exists only for them.
DOCUMENT_PARSING: dict[str, dict[str, Any]] = {
    "standard": {"label": "Current chunking", "tag": None,
                 "description": "The app default: pdf-parse text and tables, no table recognition"},
    "docling": {"label": "Docling + TableFormer", "tag": "docling",
                "description": "Heron layout finds tables and TableFormer rebuilds their cells"},
    "docling-cleanup": {"label": "Docling + TableFormer + PDF cleanups", "tag": "docling-cleanup",
                        "description": "Docling + TableFormer, plus the benchmark-only page cleanups "
                                       "(RAG_PDF_PAGE_CLEANUP): strips running headers, joins "
                                       "sentences across pages"},
}
DEFAULT_DOCUMENT_PARSING = "standard"
assert tuple(DOCUMENT_PARSING) == DOCUMENT_PARSING_MODES
Chunking = dict[str, int] | None
MAPPING_FILE = "document-id-mapping.json"
MARKER_FILE = "beir-dataset.json"


def variant_id(chunking: Chunking, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
               parsing: str = DEFAULT_DOCUMENT_PARSING) -> str:
    parts = []
    if chunking is not None:
        parts.append(f"t{chunking['maxTokens']}o{chunking['overlapTokens']}")
    for tag in (EMBEDDING_MODELS[embedding_model]["tag"], DOCUMENT_PARSING[parsing]["tag"]):
        if tag is not None:
            parts.append(tag)
    return "-".join(parts) or DEFAULT_VARIANT


def describe_chunking(chunking: Chunking) -> str:
    if chunking is None:
        return "1,200-character chunks"
    return f"{chunking['maxTokens']}-token chunks, {chunking['overlapTokens']} overlap"


def describe_variant(chunking: Chunking, embedding_model: str,
                     parsing: str = DEFAULT_DOCUMENT_PARSING) -> str:
    label = EMBEDDING_MODELS.get(embedding_model, {}).get("label", embedding_model)
    described = f"{describe_chunking(chunking)} · {label}"
    if parsing != DEFAULT_DOCUMENT_PARSING:
        described += f" · {DOCUMENT_PARSING.get(parsing, {}).get('label', parsing)}"
    return described


def runtime_id(dataset: str, chunking: Chunking,
               embedding_model: str = DEFAULT_EMBEDDING_MODEL,
               parsing: str = DEFAULT_DOCUMENT_PARSING) -> str:
    return f"runtime-{dataset}-{variant_id(chunking, embedding_model, parsing)}"


def runtime_path(dataset: str, chunking: Chunking,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 parsing: str = DEFAULT_DOCUMENT_PARSING) -> Path:
    return paths.RUNTIME_ROOT / runtime_id(dataset, chunking, embedding_model, parsing)


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
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 parsing: str = DEFAULT_DOCUMENT_PARSING) -> Path:
    runtime = runtime_path(dataset, chunking, embedding_model, parsing)
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


def is_pdf(dataset: str) -> bool:
    return is_pdf_dataset(paths.DATASETS_ROOT / dataset)


NOT_DOWNLOADED = "BEIR dataset files are not downloaded"
NO_PDFS = "Text corpus: it has no PDFs to parse"


def variant_status(dataset: str, chunking: Chunking,
                   embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                   parsing: str = DEFAULT_DOCUMENT_PARSING) -> dict[str, Any]:
    """Whether this dataset's copy for the variant is ready, being built, or missing."""
    runtime = runtime_path(dataset, chunking, embedding_model, parsing)
    status: dict[str, Any] = {"runtimeId": runtime.name, "state": "missing", "reason": None,
                              "documentCount": None, "chunkCount": None}
    if not dataset_files_present(dataset):
        status["reason"] = NOT_DOWNLOADED
        return status
    if parsing != DEFAULT_DOCUMENT_PARSING and not is_pdf(dataset):
        status["state"] = "unavailable"
        status["reason"] = NO_PDFS
        return status
    marker = read_marker(runtime)
    if marker is None:
        status["reason"] = "Not prepared for these chunking, embedding and parsing settings"
        return status
    marker_model = marker.get("embeddingModel", DEFAULT_EMBEDDING_MODEL)
    if marker_model != embedding_model:
        status["reason"] = f"Copy was embedded with {marker_model}, not {embedding_model}"
        return status
    marker_parsing = marker.get("documentParsing", DEFAULT_DOCUMENT_PARSING)
    if marker_parsing != parsing:
        status["reason"] = f"Copy was parsed with {marker_parsing}, not {parsing}"
        return status
    if marker.get("state") == "preparing":
        status["state"] = "preparing"
        status["reason"] = "Prepare job running or interrupted; prepare again to rebuild"
        return status
    mapping = mapping_path(dataset, chunking, embedding_model, parsing)
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
                 split: str = "test",
                 parsing: str = DEFAULT_DOCUMENT_PARSING) -> list[dict[str, Any]]:
    corpora = []
    for info in DATASETS:
        status = variant_status(info.name, chunking, embedding_model, parsing)
        corpora.append({
            "name": info.name,
            "description": info.description,
            "corpusSize": info.corpus_size,
            "queryCount": count_judged_queries(info.name, split),
            "splits": available_splits(info.name),
            "pdf": is_pdf(info.name),
            "prepared": status["state"] == "prepared",
            "chunking": describe_chunking(chunking),
            "variant": describe_variant(chunking, embedding_model, parsing),
            **status,
        })
    return corpora


def require_prepared(dataset: str, chunking: Chunking,
                     embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                     parsing: str = DEFAULT_DOCUMENT_PARSING) -> Path:
    """Return the mapping path for a prepared corpus variant, or raise."""
    status = variant_status(dataset, chunking, embedding_model, parsing)
    if status["state"] != "prepared":
        variant = describe_variant(chunking, embedding_model, parsing)
        raise ValueError(f"{dataset} ({variant}) is not prepared: {status['reason']}")
    return mapping_path(dataset, chunking, embedding_model, parsing)


def start_preparing(dataset: str, chunking: Chunking,
                    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                    parsing: str = DEFAULT_DOCUMENT_PARSING) -> Path:
    """Create an empty variant runtime, replacing only an unfinished earlier attempt."""
    if not dataset_files_present(dataset):
        raise ValueError(f"{dataset} dataset files are not downloaded")
    if parsing != DEFAULT_DOCUMENT_PARSING and not is_pdf(dataset):
        raise ValueError(f"{dataset} is a text corpus; {parsing} parsing applies only to PDFs")
    runtime = runtime_path(dataset, chunking, embedding_model, parsing)
    marker = read_marker(runtime)
    # A finished default copy is never overwritten by a prepare job, since earlier results
    # depend on it; deleting or regenerating it from the Corpora tab is the deliberate path.
    unfinished = marker is not None and marker.get("state") == "preparing"
    if runtime.name.endswith(f"-{DEFAULT_VARIANT}") and runtime.exists() and not unfinished:
        raise ValueError(f"{runtime.name} already exists; delete or regenerate it from the Corpora tab")
    if runtime.exists():
        if marker is None or marker.get("dataset") != dataset:
            raise ValueError(f"{runtime} exists but is not a bench corpus copy; move it aside first")
        if marker.get("state") != "preparing":
            variant = describe_variant(chunking, embedding_model, parsing)
            raise ValueError(f"{dataset} ({variant}) is already prepared")
        shutil.rmtree(runtime)
    write_marker(runtime, {"dataset": dataset, "chunking": chunking,
                           "embeddingModel": embedding_model, "documentParsing": parsing,
                           "state": "preparing"})
    return runtime


def finish_preparing(dataset: str, chunking: Chunking, embedding_model: str,
                     built_mapping: Path,
                     parsing: str = DEFAULT_DOCUMENT_PARSING) -> dict[str, Any]:
    runtime = runtime_path(dataset, chunking, embedding_model, parsing)
    target = runtime / MAPPING_FILE
    shutil.copy2(built_mapping, target)
    identity = validate_existing_runtime_schema(runtime, dataset=dataset, mapping=target)
    marker = {
        "dataset": dataset,
        "chunking": chunking,
        "embeddingModel": embedding_model,
        "documentParsing": parsing,
        "state": "prepared",
        "documentCount": identity["documentCount"],
        "chunkCount": count_chunks(runtime),
        "mappingSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    write_marker(runtime, marker)
    return marker


RUNTIME_PREFIX = "runtime-"


def folder_size(folder: Path) -> int:
    """Bytes on disk, not following the shared model-cache links."""
    return sum(path.lstat().st_size for path in folder.rglob("*")
               if path.is_file() and not path.is_symlink())


def copy_variant(marker: dict[str, Any]) -> tuple[str, Chunking, str, str]:
    return (marker["dataset"], marker.get("chunking"),
            marker.get("embeddingModel") or DEFAULT_EMBEDDING_MODEL,
            marker.get("documentParsing") or DEFAULT_DOCUMENT_PARSING)


def list_copies() -> list[dict[str, Any]]:
    """Every corpus copy on disk, whatever the saved settings; folders without a marker are skipped."""
    copies = []
    if not paths.RUNTIME_ROOT.is_dir():
        return copies
    for runtime in sorted(paths.RUNTIME_ROOT.glob(f"{RUNTIME_PREFIX}*")):
        marker = read_marker(runtime) if runtime.is_dir() else None
        if marker is None or "dataset" not in marker:
            continue
        dataset, chunking, model, parsing = copy_variant(marker)
        copies.append({
            "runtimeId": runtime.name,
            "dataset": dataset,
            "chunking": describe_chunking(chunking),
            "embeddingModel": model,
            "documentParsing": parsing,
            "variant": describe_variant(chunking, model, parsing),
            "state": marker.get("state", "prepared"),
            "documentCount": marker.get("documentCount"),
            "chunkCount": marker.get("chunkCount"),
            "sizeBytes": folder_size(runtime),
            "default": runtime.name.endswith(f"-{DEFAULT_VARIANT}"),
            "regenerable": runtime.name == runtime_id(dataset, chunking, model, parsing),
        })
    return copies


def copy_path(runtime_name: str) -> Path:
    """The folder of a corpus copy named by the GUI, refusing anything that is not one."""
    if (not runtime_name.startswith(RUNTIME_PREFIX) or "/" in runtime_name
            or "\\" in runtime_name or runtime_name in (".", "..")):
        raise ValueError(f"{runtime_name!r} is not a corpus copy name")
    runtime = paths.RUNTIME_ROOT / runtime_name
    if runtime.parent != paths.RUNTIME_ROOT or runtime.is_symlink() or not runtime.is_dir():
        raise FileNotFoundError(f"{runtime_name} does not exist")
    if read_marker(runtime) is None:
        raise ValueError(f"{runtime_name} has no {MARKER_FILE}; it is not a bench corpus copy")
    return runtime


def copy_marker(runtime_name: str) -> dict[str, Any]:
    marker = read_marker(copy_path(runtime_name))
    assert marker is not None
    return marker


def delete_copy(runtime_name: str) -> Path:
    """Remove a corpus copy; links to the shared model caches are removed, not their targets."""
    runtime = copy_path(runtime_name)
    shutil.rmtree(runtime)
    return runtime
