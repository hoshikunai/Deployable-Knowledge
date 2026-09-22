from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import socket
import signal
import sqlite3
import hashlib
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
sys.path.insert(0, str(REPOSITORY_ROOT))
from benchmarks.benchmark_lease import (terminate_group, MemoryWatchdog, memory_preflight,
    BenchmarkLease, TimedOutError, MemoryPausedError)
from benchmarks.benchmark_lease import start_identity, atomic_json


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def select_available_port(preferred: int) -> int:
    if port_is_available(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


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


def prepare_runtime(dataset: str, share_model_cache: bool, runtime_id: str | None = None) -> Path:
    runtime = RUNTIME_ROOT / (runtime_id or f"runtime-{dataset}-001")
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


def validate_existing_runtime_schema(runtime: Path, dataset: str | None = None,
                                     mapping: Path | None = None) -> dict[str, Any]:
    marker = runtime / "beir-dataset.json"
    if dataset is not None:
        if not marker.is_file() or json.loads(marker.read_text()).get("dataset") != dataset:
            raise RuntimeError("Existing runtime dataset marker mismatch")
    database_path = runtime / "app.db"
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
        existing_tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {
            "__drizzle_migrations",
            "app_state",
            "documents",
            "document_chunks",
        }
        if not required_tables.issubset(existing_tables):
            raise RuntimeError(
                "Existing benchmark database is missing required tables: "
                f"{sorted(required_tables - existing_tables)}"
            )
        migration_count = database.execute(
            "SELECT COUNT(*) FROM __drizzle_migrations"
        ).fetchone()[0]
        if migration_count < 1:
            raise RuntimeError("Existing benchmark database has no applied migration")
        columns = {row[1] for row in database.execute("PRAGMA table_info(__drizzle_migrations)")}
        migration_column = "hash" if "hash" in columns else None
        migrations = database.execute(f"SELECT {migration_column} FROM __drizzle_migrations ORDER BY id").fetchall() if migration_column else database.execute("SELECT * FROM __drizzle_migrations ORDER BY id").fetchall()
        migration_hash = hashlib.sha256(json.dumps(migrations).encode()).hexdigest()
        if mapping is not None:
            value = json.loads(mapping.read_text())
            expected = set(value.get("applicationToBeir", {}))
            actual = {row[0] for row in database.execute("SELECT id FROM documents")}
            if actual != expected:
                raise RuntimeError("Existing runtime document IDs do not match mapping")
    ids_hash = hashlib.sha256(json.dumps(sorted(actual)).encode()).hexdigest() if mapping is not None else None
    return {"dataset": dataset, "migrationHash": migration_hash, "documentCount": len(actual) if mapping is not None else None, "documentIdsHash": ids_hash}


def start_server(
    runtime: Path,
    base_url: str,
    port: int,
    startup_timeout: int,
    reuse_existing_schema: bool = False,
    dataset: str | None = None,
    mapping: Path | None = None,
) -> tuple[subprocess.Popen[bytes], BinaryIO, Path]:
    if heartbeat(base_url):
        raise RuntimeError(
            f"A server is already responding at {base_url}. Stop it before "
            "running the isolated suite."
        )

    if reuse_existing_schema:
        validate_existing_runtime_schema(runtime, dataset=dataset, mapping=mapping)

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
    if reuse_existing_schema:
        environment.pop("DK_MIGRATIONS_DIR", None)
    process = subprocess.Popen(
        ["node", str(SERVER_ENTRYPOINT)],
        cwd=runtime,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    marker = runtime / 'benchmark-owner.json'
    atomic_json(marker, {'pid':process.pid,'startIdentity':start_identity(process.pid),'pgid':process.pid,'command':'node '+str(SERVER_ENTRYPOINT),'runtimePath':str(runtime.resolve()),'port':port,'heartbeatAt':time.time()})
    process._benchmark_marker = marker
    deadline = time.monotonic() + startup_timeout

    try:
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

        raise RuntimeError(
            f"The server did not become ready within {startup_timeout}s.\n"
            f"{tail(log_path)}"
        )
    except BaseException:
        terminate_group(process.pid)
        process.wait(timeout=15)
        marker.unlink(missing_ok=True)
        log_file.close()
        raise


def stop_server(
    process: subprocess.Popen[bytes],
    log_file: BinaryIO,
) -> None:
    if process.poll() is None:
        terminate_group(process.pid)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    marker = getattr(process, '_benchmark_marker', None)
    if marker and marker.exists(): marker.unlink()
    log_file.close()


def run_dataset(
    dataset: str,
    arguments: argparse.Namespace,
    suite_id: str,
    lease_owner: BenchmarkLease | None = None,
) -> dict[str, Any]:
    if lease_owner is None:
        memory_preflight()
        with BenchmarkLease(runName=suite_id, dataset=dataset, port=arguments.port) as owner:
            return run_dataset(dataset, arguments, suite_id, owner)
    if lease_owner.fd is None or lease_owner.metadata.get('pid') != os.getpid():
        raise RuntimeError('invalid lease owner')
    download_dataset(dataset, DATASETS_ROOT)
    runtime = prepare_runtime(
        dataset,
        share_model_cache=not arguments.no_shared_model_cache,
        runtime_id=getattr(arguments, "runtime_id", None),
    )
    runtime_identity = validate_existing_runtime_schema(
        runtime, dataset=dataset,
        mapping=getattr(arguments, "reuse_document_mapping", None)
        if getattr(arguments, "reuse_existing_runtime", False) else None,
    ) if getattr(arguments, "reuse_existing_runtime", False) else None
    port = select_available_port(arguments.port)
    base_url = f"http://127.0.0.1:{port}"
    process, log_file, log_path = start_server(
        runtime,
        base_url,
        port,
        arguments.startup_timeout,
        reuse_existing_schema=getattr(arguments, "reuse_existing_schema", False),
        dataset=dataset,
        mapping=getattr(arguments, "reuse_document_mapping", None)
        if getattr(arguments, "reuse_existing_runtime", False) else None,
    )
    run_name = getattr(arguments, "run_name", None) or f"{suite_id}-{arguments.split}"
    run_directory = RUNS_ROOT / f"{dataset}-{run_name}"
    child: subprocess.Popen[bytes] | None = None

    try:
        command = [
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
            ]
        if getattr(arguments, "reuse_existing_runtime", False):
            command += ["--reuse-existing-runtime", "--reuse-document-mapping",
                        str(arguments.reuse_document_mapping), "--runtime-fingerprint",
                        hashlib.sha256(json.dumps(runtime_identity, sort_keys=True).encode()).hexdigest()]
        if getattr(arguments, "resume", False):
            command.append("--resume")
        child = subprocess.Popen(command, cwd=REPOSITORY_ROOT, start_new_session=True)
        resume_command = (f"python benchmarks/rag-evaluation/full_beir_plan.py --execute --resume "
                          f"--dataset {dataset} --run-name {run_name} "
                          f"--search-depth {arguments.search_depth} "
                          f"--timeout-minutes {getattr(arguments, 'timeout_seconds', 2700) // 60}")
        if getattr(arguments, "reuse_existing_runtime", False):
            resume_command += f" --reuse-existing-runtime --reuse-document-mapping {arguments.reuse_document_mapping}"
        supervisor_directory = RUNS_ROOT / f".supervisor-{dataset}-{run_name}"
        watchdog = MemoryWatchdog([process.pid, child.pid], ceiling=getattr(arguments, "memory_ceiling", 7*1024**3), warning=getattr(arguments, "memory_warning", 6*1024**3), resume_command=resume_command, status_path=supervisor_directory/'status.json', samples_path=supervisor_directory/'memory-samples.jsonl')
        deadline = time.monotonic() + getattr(arguments, "timeout_seconds", 10**9)
        while child.poll() is None:
            if watchdog.poll() == "memory-paused":
                terminate_group(child.pid)
                child.wait(timeout=15)
                raise MemoryPausedError("memory-paused", peak_owned_rss=watchdog.peak,
                    last_sample=watchdog.last, resume_command=watchdog.resume_command)
            if time.monotonic() >= deadline:
                terminate_group(child.pid)
                child.wait(timeout=15)
                raise TimedOutError("benchmark timed out", peak_owned_rss=watchdog.peak,
                    last_sample=watchdog.last, resume_command=watchdog.resume_command)
            time.sleep(min(watchdog.interval, .5))
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
            raise
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
    finally:
        if child is not None and child.poll() is None:
            terminate_group(child.pid)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=15)
        stop_server(process, log_file)

    return {
        "dataset": dataset,
        "runtime": str(runtime.relative_to(REPOSITORY_ROOT)),
        "serverLog": str(log_path.relative_to(REPOSITORY_ROOT)),
        "runDirectory": str(run_directory.relative_to(REPOSITORY_ROOT)),
        "port": port,
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
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }

    def interrupted(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"benchmark suite interrupted by signal {signum}")

    for signum in previous_handlers:
        signal.signal(signum, interrupted)

    try:
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
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
