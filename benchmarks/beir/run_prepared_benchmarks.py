"""Run prepared BEIR runtimes sequentially with bounded, resumable sessions.

This entry point never ingests a corpus. It only uses an existing database and
its saved document-ID mapping, and leaves failed runs in place for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_suite import validate_existing_runtime_schema


ROOT = Path(__file__).resolve().parents[2]
BEIR_ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "benchmarks" / "rag-evaluation" / "full_beir_plan.py"
PREPARED_MAPPINGS = {
    "arguana": "benchmarks/beir/runs/arguana-suite-20260910-153730-040144-test/document-id-mapping.json",
    "nfcorpus": "benchmarks/beir/runs/nfcorpus-nfcorpus-full-20260915-174940-224808/document-id-mapping.json",
    "scifact": "benchmarks/beir/runs/scifact-scifact-full-test-v2/document-id-mapping.json",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*", choices=sorted(PREPARED_MAPPINGS),
        help="Prepared datasets to run, in order (default: arguana).",
    )
    parser.add_argument(
        "--mapping", action="append", default=[], metavar="DATASET=PATH",
        help="Override a saved mapping path; repeat for multiple datasets.",
    )
    parser.add_argument("--run-name", help="Stable run name; only for one dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume --run-name.")
    parser.add_argument("--timeout-minutes", type=int, default=45)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument(
        "--document-depth", type=int, default=10,
        help="Evaluate up to this many unique documents per query (10-100).",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and show commands only.")
    return parser.parse_args()


def mapping_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or dataset not in PREPARED_MAPPINGS or not path:
            raise ValueError("--mapping must be DATASET=PATH for a prepared dataset")
        if dataset in overrides:
            raise ValueError(f"Mapping specified twice for {dataset}")
        overrides[dataset] = path
    return overrides


def preflight_dataset(dataset: str, mapping_argument: str) -> dict[str, Any]:
    mapping = Path(mapping_argument)
    if not mapping.is_absolute():
        mapping = ROOT / mapping
    if not mapping.is_file():
        raise FileNotFoundError(f"Saved {dataset} mapping is missing: {mapping}")
    dataset_root = BEIR_ROOT / "datasets" / dataset
    for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        if not (dataset_root / relative).is_file():
            raise FileNotFoundError(
                f"Prepared {dataset} dataset is missing {relative}; no download or ingestion was attempted"
            )
    runtime = ROOT / ".cache" / "beir" / f"runtime-{dataset}-001"
    return validate_existing_runtime_schema(runtime, dataset=dataset, mapping=mapping)


def plan_command(dataset: str, mapping: str, run_name: str,
                 timeout_minutes: int, resume: bool,
                 document_depth: int = 10) -> list[str]:
    command = [
        sys.executable, str(PLAN), "--execute", "--dataset", dataset,
        "--split", "test", "--reuse-existing-runtime",
        "--reuse-document-mapping", mapping,
        "--timeout-minutes", str(timeout_minutes), "--run-name", run_name,
        "--search-depth", str(document_depth),
    ]
    if resume:
        command.append("--resume")
    return command


def read_status(run_name: str) -> dict[str, Any]:
    path = ROOT / "benchmarks" / "rag-evaluation" / "runs" / run_name / "status.json"
    if not path.is_file():
        raise RuntimeError(f"Supervisor did not write status: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid supervisor status: {path}")
    return value


def finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_metrics(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return False


def verify_completed_run(dataset: str, run_name: str, document_depth: int = 10) -> Path:
    run_directory = BEIR_ROOT / "runs" / f"{dataset}-{run_name}"
    selection = json.loads((run_directory / "query-selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run_directory / "run-summary.json").read_text(encoding="utf-8"))
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    failures = json.loads((run_directory / "failures.json").read_text(encoding="utf-8"))
    query_ids = selection["queryIds"]
    with (BEIR_ROOT / "datasets" / dataset / "qrels" / "test.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        judged_query_ids = {row["query-id"] for row in csv.DictReader(stream, delimiter="\t")}
    if (summary["queryCount"] != len(query_ids)
            or set(query_ids) != judged_query_ids or failures):
        raise RuntimeError(f"Completed {dataset} artifacts have missing queries or failures")
    if set(metrics) != {"bm25", "semantic", "hybrid"} or not finite_metrics(metrics):
        raise RuntimeError(f"Completed {dataset} metrics are incomplete or non-finite")
    for method in metrics:
        if f"Recall@{document_depth}" not in metrics[method].get("recall", {}):
            raise RuntimeError(
                f"Completed {dataset} metrics lack Recall@{document_depth} for {method}"
            )
    return metrics_path


def run_command(command: list[str]) -> int:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Benchmark is still stopping; inspect the lease before retrying", file=sys.stderr)
        raise


def run_dataset(dataset: str, mapping: str, run_name: str,
                timeout_minutes: int, max_sessions: int, resume: bool,
                document_depth: int = 10) -> Path:
    previous_checkpoints = 0
    if resume:
        saved = read_status(run_name)
        if saved.get("status") == "completed":
            return verify_completed_run(dataset, run_name, document_depth)
        run_config = BEIR_ROOT / "runs" / f"{dataset}-{run_name}" / "run-config.json"
        if not run_config.is_file():
            raise RuntimeError(
                f"{run_name} stopped before creating a resumable BEIR run; use a new run name"
            )
        previous_checkpoints = int(saved.get("checkpointCount", 0))

    for session in range(1, max_sessions + 1):
        from benchmarks.benchmark_lease import memory_preflight

        memory_preflight()
        command = plan_command(
            dataset, mapping, run_name, timeout_minutes, resume, document_depth,
        )
        print(f"{dataset}: session {session}/{max_sessions} — {' '.join(command)}", flush=True)
        return_code = run_command(command)
        try:
            status = read_status(run_name)
        except RuntimeError as error:
            if return_code != 0:
                raise RuntimeError(
                    f"{dataset} supervisor exited {return_code} before writing status; "
                    "see the error printed above"
                ) from error
            raise
        state = status.get("status")
        if state == "completed" and return_code == 0:
            metrics_path = verify_completed_run(dataset, run_name, document_depth)
            print(f"{dataset}: complete; metrics: {metrics_path}", flush=True)
            return metrics_path
        checkpoints = int(status.get("checkpointCount", 0))
        if state != "timed-out" or checkpoints <= previous_checkpoints:
            raise RuntimeError(
                f"{dataset} stopped ({state}, exit {return_code}, {checkpoints} checkpoints). "
                f"Inspect benchmarks/rag-evaluation/runs/{run_name}/status.json; no automatic retry."
            )
        print(f"{dataset}: timed out safely after {checkpoints} checkpoints; resuming", flush=True)
        previous_checkpoints = checkpoints
        resume = True
    raise RuntimeError(
        f"{dataset} reached {max_sessions} bounded sessions; resume with "
        f"--resume --run-name {run_name} {dataset}"
    )


def main() -> None:
    arguments = parse_arguments()
    datasets = arguments.datasets or ["arguana"]
    if arguments.timeout_minutes < 1 or arguments.max_sessions < 1:
        raise ValueError("Timeout and max sessions must both be positive")
    if not 10 <= arguments.document_depth <= 100:
        raise ValueError("--document-depth must be between 10 and 100")
    if arguments.run_name and len(datasets) != 1:
        raise ValueError("--run-name is only valid for one dataset")
    if arguments.resume and not arguments.run_name:
        raise ValueError("--resume requires --run-name")
    if len(set(datasets)) != len(datasets):
        raise ValueError("List each dataset only once")
    overrides = mapping_overrides(arguments.mapping)
    mappings = {dataset: overrides.get(dataset, PREPARED_MAPPINGS[dataset]) for dataset in datasets}
    for dataset, mapping in mappings.items():
        identity = preflight_dataset(dataset, mapping)
        print(f"{dataset}: prepared runtime validated ({identity['documentCount']} documents)")
    if not arguments.skip_build and not arguments.dry_run:
        from benchmarks.benchmark_lease import BenchmarkLease, memory_preflight

        memory_preflight()
        with BenchmarkLease(runName="beir-build", dataset="preflight"):
            subprocess.run(["npm", "run", "build:electron"], cwd=ROOT, check=True)
    if not (ROOT / "build" / "index.js").is_file() and not arguments.dry_run:
        raise RuntimeError("Application server build is missing; omit --skip-build")
    for dataset, mapping in mappings.items():
        run_name = arguments.run_name or (
            f"{dataset}-full-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
        )
        if arguments.dry_run:
            print("DRY RUN:", " ".join(plan_command(
                dataset, mapping, run_name, arguments.timeout_minutes, arguments.resume,
                arguments.document_depth,
            )))
            continue
        run_dataset(
            dataset, mapping, run_name, arguments.timeout_minutes,
            arguments.max_sessions, arguments.resume,
            arguments.document_depth,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
