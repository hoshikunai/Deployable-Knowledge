"""Break a finished run's metrics down by query group.

Datasets built with a query-groups.json (such as build_openrag_dataset.py's) tag each query
with its source modality and type. This re-scores a run's saved document rankings per group.

    benchmarks/beir/.venv/bin/python benchmarks/beir/group_metrics.py \\
        benchmarks/beir/runs/openrag-arxiv-RUN --by source
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader

from run import DATASETS_ROOT, METHODS, evaluate_method

QUERY_GROUPS = "query-groups.json"
REPORTED = (("ndcg", "NDCG@10"), ("mrr", "MRR@10"), ("recall", "Recall@10"))


def group_metrics(run_directory: Path, by: str) -> dict[str, dict[str, dict[str, float]]]:
    """{group: {method: {metric: value}}} plus the group's query count under "queries"."""
    config = json.loads((run_directory / "run-config.json").read_text(encoding="utf-8"))
    dataset_path = DATASETS_ROOT / config["dataset"]
    groups_path = dataset_path / QUERY_GROUPS
    if not groups_path.is_file():
        raise SystemExit(f"{config['dataset']} has no {QUERY_GROUPS}")
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    rankings = json.loads((run_directory / "document-rankings.json").read_text(encoding="utf-8"))
    _, _, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=config["split"])

    members: dict[str, list[str]] = defaultdict(list)
    for query_id in rankings[METHODS[0]]:
        members[groups[query_id][by]].append(query_id)

    report = {}
    for group, query_ids in sorted(members.items()):
        group_qrels = {query_id: qrels[query_id] for query_id in query_ids}
        report[group] = {"queries": {"count": len(query_ids)}}
        for method in METHODS:
            scores = evaluate_method(
                group_qrels, {query_id: rankings[method][query_id] for query_id in query_ids})
            report[group][method] = {label: scores[family][label] for family, label in REPORTED}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--by", choices=("source", "type"), default="source")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    arguments = parser.parse_args()
    report = group_metrics(arguments.run_directory, arguments.by)
    if arguments.json:
        print(json.dumps(report, indent=2))
        return
    labels = [label for _, label in REPORTED]
    print(f"{arguments.by:<18} {'queries':>7}  {'method':<9} " + " ".join(f"{label:>10}" for label in labels))
    for group, methods in report.items():
        for method in METHODS:
            values = " ".join(f"{methods[method][label]:>10.4f}" for label in labels)
            print(f"{group:<18} {methods['queries']['count']:>7}  {method:<9} {values}")


if __name__ == "__main__":
    main()
