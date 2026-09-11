from __future__ import annotations

import argparse
from pathlib import Path

from dataset_catalog import DATASETS, download_dataset


HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract public BEIR datasets.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="BEIR dataset names, for example nfcorpus arguana.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recommended datasets without downloading them.",
    )
    return parser.parse_args()


def print_catalog() -> None:
    for dataset in DATASETS:
        splits = ",".join(dataset.splits)
        print(
            f"{dataset.name:12} corpus={dataset.corpus_size:>5} "
            f"test_queries={dataset.test_queries:>5} "
            f"splits={splits:14} {dataset.description}"
        )


def main() -> None:
    arguments = parse_arguments()

    if arguments.list:
        print_catalog()

    if not arguments.datasets:
        if not arguments.list:
            print_catalog()
            print("\nPass one or more dataset names to download them.")
        return

    for dataset in arguments.datasets:
        print(f"Downloading {dataset}...")
        path = download_dataset(dataset, DATASETS_ROOT)
        print(f"Installed {dataset} at {path}")


if __name__ == "__main__":
    main()
