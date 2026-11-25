from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beir import util


BEIR_DATASET_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets"
)
DATASET_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# PDF datasets (built by build_pdf_dataset.py) carry this manifest; their corpus.jsonl lists
# one entry per page, and the PDFs themselves are uploaded through the app's file ingestion.
PDF_MANIFEST = "pdf-manifest.json"
# How PDFs are parsed at ingestion: "standard" is the app default (pdf-parse tables);
# "docling" turns on table recognition (Heron layout + TableFormer). OCR is the same for both.
DOCUMENT_PARSING_MODES = ("standard", "docling", "docling-cleanup")
TABLE_RECOGNITION_MODES = ("docling", "docling-cleanup")
# Where llama.cpp embeds, mapped to the app's embedding device setting. "vulkan" is the app's
# "gpu", which refuses to fall back to the CPU.
EMBEDDING_DEVICE_SETTINGS = {"auto": "auto", "cpu": "cpu", "vulkan": "gpu"}
# A PDF manifest may score some qrel splits by page and others by whole document.
RANKING_UNITS = ("page", "document")
# Qrel splits the harness accepts; "doc" is the whole-document split of a PDF dataset.
QUERY_SPLITS = ("test", "train", "dev", "doc")


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    splits: tuple[str, ...]
    corpus_size: str
    test_queries: str


DATASETS = (
    DatasetInfo(
        name="scifact",
        description="Scientific claim-to-evidence retrieval",
        splits=("train", "test"),
        corpus_size="5K",
        test_queries="300",
    ),
    DatasetInfo(
        # Built locally by dedup_experiment/make_scifact_dup.py; not downloadable.
        name="scifact-dup",
        description="SciFact with duplicated relevant documents (dedup experiment)",
        splits=("test",),
        corpus_size="5.5K",
        test_queries="300",
    ),
    DatasetInfo(
        name="nfcorpus",
        description="Biomedical and nutrition information retrieval",
        splits=("train", "dev", "test"),
        corpus_size="3.6K",
        test_queries="323",
    ),
    DatasetInfo(
        name="arguana",
        description="Counterargument retrieval",
        splits=("test",),
        corpus_size="8.67K",
        test_queries="1,406",
    ),
    DatasetInfo(
        name="scidocs",
        description="Scientific document citation retrieval",
        splits=("test",),
        corpus_size="25K",
        test_queries="1,000",
    ),
    DatasetInfo(
        name="fiqa",
        description="Financial question-to-answer retrieval",
        splits=("train", "dev", "test"),
        corpus_size="57K",
        test_queries="648",
    ),
    DatasetInfo(
        # Built locally by build_pdf_dataset.py from the app's PDF eval set; not downloadable.
        name="pdf-eval",
        description="Air Force publications as PDFs, scored by page (X queries target tables)",
        splits=("test",),
        corpus_size="8 PDFs, 1,071 pages",
        test_queries="81",
    ),
    DatasetInfo(
        # Built locally by build_openrag_dataset.py from Vectara's Open RAG Benchmark (arXiv);
        # "test" scores pages of the aligned section, "doc" scores whole papers.
        name="openrag-arxiv",
        description="arXiv papers as PDFs (Open RAG Benchmark); test scores pages, doc scores papers",
        splits=("test", "doc"),
        corpus_size="250 PDFs (default subset)",
        test_queries="~700",
    ),
    DatasetInfo(
        # Built locally by build_ohr_bench_dataset.py from OHR-Bench (finance, academic, manual
        # and administration by default); query groups label table evidence and scanned pages.
        name="ohr-bench",
        description="OHR-Bench PDFs in four table-heavy domains, scored by page (1,909 table queries)",
        splits=("test", "doc"),
        corpus_size="383 PDFs, 6,209 pages",
        test_queries="5,029",
    ),
    DatasetInfo(
        # Built locally by build_financebench_dataset.py from FinanceBench's open-source sample.
        name="financebench",
        description="SEC filings as PDFs (FinanceBench), mostly financial-table questions, scored by page",
        splits=("test", "doc"),
        corpus_size="84 PDFs, 12,013 pages",
        test_queries="150",
    ),
    DatasetInfo(
        # Built locally by build_mmlongbench_dataset.py from MMLongBench-Doc.
        name="mmlongbench-doc",
        description="Long mixed PDFs (MMLongBench-Doc), scored by page (209 table queries)",
        splits=("test", "doc"),
        corpus_size="132 PDFs, 6,252 pages",
        test_queries="825",
    ),
)


def validate_dataset_name(dataset: str) -> str:
    if not DATASET_NAME_PATTERN.fullmatch(dataset):
        raise ValueError(f"Invalid BEIR dataset name: {dataset}")
    return dataset


def dataset_url(dataset: str) -> str:
    name = validate_dataset_name(dataset)
    return f"{BEIR_DATASET_BASE_URL}/{name}.zip"


def load_pdf_manifest(dataset_path: Path) -> dict[str, Any] | None:
    path = dataset_path / PDF_MANIFEST
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents or not all(
        isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("file"), str)
        and isinstance(item.get("pages"), int)
        and item["pages"] > 0
        for item in documents
    ):
        raise RuntimeError(f"Invalid PDF manifest: {path}")
    split_units = manifest.get("splitUnits", {})
    if not isinstance(split_units, dict) or not all(
        isinstance(split, str) and unit in RANKING_UNITS for split, unit in split_units.items()
    ):
        raise RuntimeError(f"Invalid splitUnits in PDF manifest: {path}")
    return manifest


def split_ranking_unit(manifest: dict[str, Any] | None, split: str) -> str:
    """What a split's qrels judge: a text corpus's documents, or a PDF's pages unless the
    manifest names the split a whole-document split."""
    if manifest is None:
        return "document"
    return manifest.get("splitUnits", {}).get(split, "page")


def is_pdf_dataset(dataset_path: Path) -> bool:
    return (dataset_path / PDF_MANIFEST).is_file()


def download_dataset(dataset: str, datasets_root: Path) -> Path:
    # Datasets already on disk (including locally built ones such as scifact-dup) are used
    # as they are; BEIR's helper would otherwise try to download a missing archive.
    local = datasets_root / dataset
    if all((local / name).is_file() for name in ("corpus.jsonl", "queries.jsonl")):
        return local
    datasets_root.mkdir(parents=True, exist_ok=True)
    downloaded_path = util.download_and_unzip(
        dataset_url(dataset),
        str(datasets_root),
    )
    return Path(downloaded_path)
