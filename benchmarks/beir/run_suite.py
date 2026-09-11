from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib import error, request

from dataset_catalog import DATASETS, download_dataset, validate_dataset_name


HARNESS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = HARNESS_ROOT.parent.parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
RUNS_ROOT = HARNESS_ROOT / "runs"
RUNTIME_ROOT = REPOSITORY_ROOT / ".cache" / "beir"
SERVER_ENTRYPOINT = REPOSITORY_ROOT / "build" / "index.js"
MIGRATIONS_ROOT = REPOSITORY_ROOT / "drizzle"
SHARED_MODEL_CACHE = REPOSITORY_ROOT / ".cache" / "transformersjs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic query-count sweeps against isolated BEIR "
            "dataset databases."
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="BEIR dataset names, for example scifact nfcorpus.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--query-counts",
        default="10,50,100,all",
        help="Nested query checkpoints (default: 10,50,100,all).",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=10,
        help="Chunks requested per retrieval method (default: 10).",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=4179)
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=120,
        help="Seconds to wait for each TypeScript server.",
    )
    parser.add_argument(
        "--no-shared-model-cache",
        action="store_true",
        help="Do not link the repository Transformers.js cache.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> list[str]:
    if arguments.split not in {"train", "dev", "test"}:
        raise ValueError("--split must be train, dev, or test")
    if arguments.search_depth < 10:
        raise ValueError("--search-depth must be at least 10")
    if not 1 <= arguments.port <= 65_535:
        raise ValueError("--port must be between 1 and 65535")

    datasets = [validate_dataset_name(value) for value in arguments.datasets]
    if len(set(datasets)) != len(datasets):
        raise ValueError("Each dataset may only appear once in a suite")

    catalog = {dataset.name: dataset for dataset in DATASETS}
    for dataset in datasets:
        info = catalog.get(dataset)
        if info and arguments.split not in info.splits:
            supported = ", ".join(info.splits)
            raise ValueError(
                f"{dataset} has no {arguments.split} split; use {supported}"
            )

    return datasets


def ensure_build_exists() -> None:
    if not SERVER_ENTRYPOINT.is_file():
        raise RuntimeError(
            "The application server build is missing. Run "
            "'npm run build:electron' from the repository root first."
        )


def prepare_runtime(dataset: str, share_model_cache: bool) -> Path:
    runtime = RUNTIME_ROOT / f"runtime-{dataset}-001"
    marker_path = runtime / "beir-dataset.json"
    database_path = runtime / "app.db"

    runtime.mkdir(parents=True, exist_ok=True)

    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("dataset") != dataset:
            raise RuntimeError(
                f"Runtime {runtime} belongs to {marker.get('dataset')}"
            )
    elif database_path.exists():
        raise RuntimeError(
            f"Refusing to reuse unlabelled database {database_path}. "
            "Move it aside or use a clean dataset-specific runtime."
        )
    else:
        marker_path.write_text(
            json.dumps({"dataset": dataset}, indent=2) + "\n",
            encoding="utf-8",
        )

    if share_model_cache and SHARED_MODEL_CACHE.is_dir():
        cache_parent = runtime / ".cache"
        cache_parent.mkdir(parents=True, exist_ok=True)
        runtime_cache = cache_parent / "transformersjs"
        if not runtime_cache.exists() and not runtime_cache.is_symlink():
            runtime_cache.symlink_to(
                SHARED_MODEL_CACHE.resolve(),
                target_is_directory=True,
            )

    return runtime


def heartbeat(base_url: str, timeout: float = 2) -> bool:
    try:
        with request.urlopen(f"{base_url}/heartbeat", timeout=timeout):
            return True
    except (error.URLError, TimeoutError):
        return False


def tail(path: Path, line_count: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        .splitlines()[-line_count:]
    )


def start_server(
    runtime: Path,
    base_url: str,
    port: int,
    startup_timeout: int,
) -> tuple[subprocess.Popen[bytes], BinaryIO, Path]:
    if heartbeat(base_url):
        raise RuntimeError(
            f"A server is already responding at {base_url}. Stop it before "
            "running the isolated suite."
        )

    log_path = runtime / "server.log"
    log_file = log_path.open("ab")
    environment = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "ORIGIN": base_url,
        "BODY_SIZE_LIMIT": "Infinity",
        "DK_MIGRATIONS_DIR": str(MIGRATIONS_ROOT),
    }
    process = subprocess.Popen(
        ["node", str(SERVER_ENTRYPOINT)],
        cwd=runtime,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + startup_timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"The server exited with status {process.returncode}.\n"
                f"{tail(log_path)}"
            )
        if heartbeat(base_url):
            return process, log_file, log_path
        time.sleep(0.5)

    process.terminate()
    process.wait(timeout=15)
    log_file.flush()
    raise RuntimeError(
        f"The server did not become ready within {startup_timeout}s.\n"
        f"{tail(log_path)}"
    )


def stop_server(
    process: subprocess.Popen[bytes],
    log_file: BinaryIO,
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    log_file.close()


def run_dataset(
    dataset: str,
    arguments: argparse.Namespace,
    suite_id: str,
) -> dict[str, Any]:
    download_dataset(dataset, DATASETS_ROOT)
    runtime = prepare_runtime(
        dataset,
        share_model_cache=not arguments.no_shared_model_cache,
    )
    base_url = f"http://127.0.0.1:{arguments.port}"
    process, log_file, log_path = start_server(
        runtime,
        base_url,
        arguments.port,
        arguments.startup_timeout,
    )
    run_name = f"{suite_id}-{arguments.split}"
    run_directory = RUNS_ROOT / f"{dataset}-{run_name}"

    try:
        subprocess.run(
            [
                sys.executable,
                str(HARNESS_ROOT / "run.py"),
                "--dataset",
                dataset,
                "--split",
                arguments.split,
                "--base-url",
                base_url,
                "--search-depth",
                str(arguments.search_depth),
                "--query-counts",
                arguments.query_counts,
                "--sample-seed",
                str(arguments.sample_seed),
                "--run-name",
                run_name,
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
    finally:
        stop_server(process, log_file)

    return {
        "dataset": dataset,
        "runtime": str(runtime.relative_to(REPOSITORY_ROOT)),
        "serverLog": str(log_path.relative_to(REPOSITORY_ROOT)),
        "runDirectory": str(run_directory.relative_to(REPOSITORY_ROOT)),
        "metricsByQueryCount": json.loads(
            (run_directory / "metrics-by-query-count.json").read_text(
                encoding="utf-8"
            )
        ),
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nSuite summary (largest query checkpoint)")
    print("dataset      queries method     NDCG@10 Recall@10")
    for result in results:
        metrics_by_count = result["metricsByQueryCount"]
        largest_count = max(int(value) for value in metrics_by_count)
        metrics = metrics_by_count[str(largest_count)]
        for method in ("bm25", "semantic", "hybrid"):
            ndcg = metrics[method]["ndcg"]["NDCG@10"]
            recall = metrics[method]["recall"]["Recall@10"]
            print(
                f"{result['dataset']:12} {largest_count:7} "
                f"{method:10} {ndcg:7.4f} {recall:9.4f}"
            )


def main() -> None:
    arguments = parse_arguments()
    datasets = validate_arguments(arguments)
    ensure_build_exists()

    suite_id = datetime.now().strftime("suite-%Y%m%d-%H%M%S-%f")
    suite_directory = RUNS_ROOT / "suites" / suite_id
    suite_directory.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        result = run_dataset(dataset, arguments, suite_id)
        results.append(result)

    summary = {
        "suiteId": suite_id,
        "split": arguments.split,
        "queryCounts": arguments.query_counts,
        "sampleSeed": arguments.sample_seed,
        "searchDepth": arguments.search_depth,
        "results": results,
    }
    summary_path = suite_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_summary(results)
    print(f"\nSuite summary written to {summary_path}")


if __name__ == "__main__":
    main()
