from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from beir import util


BEIR_DATASET_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets"
)
DATASET_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


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
)


def validate_dataset_name(dataset: str) -> str:
    if not DATASET_NAME_PATTERN.fullmatch(dataset):
        raise ValueError(f"Invalid BEIR dataset name: {dataset}")
    return dataset


def dataset_url(dataset: str) -> str:
    name = validate_dataset_name(dataset)
    return f"{BEIR_DATASET_BASE_URL}/{name}.zip"


def download_dataset(dataset: str, datasets_root: Path) -> Path:
    datasets_root.mkdir(parents=True, exist_ok=True)
    downloaded_path = util.download_and_unzip(
        dataset_url(dataset),
        str(datasets_root),
    )
    return Path(downloaded_path)
