"""Run one HAKARI job under the repository-wide benchmark safeguards."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.benchmark_lease import (  # noqa: E402
    BenchmarkLease,
    InterruptedError,
    MemoryPausedError,
    MemoryWatchdog,
    SupervisionError,
    TimedOutError,
    atomic_json,
    memory_preflight,
    terminate_group,
    validate_status,
)

DEFAULT_STATUS_PATH = REPOSITORY_ROOT / ".cache" / "hakari" / "status.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a HAKARI script with exclusive ownership and memory limits."
    )
    parser.add_argument("--lease-path")
    parser.add_argument("--meta-path")
    parser.add_argument("--status-path")
    parser.add_argument("--samples-path")
    parser.add_argument("--skip-memory-preflight", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=3_600)
    parser.add_argument("--memory-ceiling", type=int, default=7 * 1024**3)
    parser.add_argument("--memory-warning", type=int, default=6 * 1024**3)
    parser.add_argument("--supervise-command", nargs=argparse.REMAINDER)
    parser.add_argument("script", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def status_paths(arguments: argparse.Namespace) -> tuple[Path, Path]:
    if arguments.status_path:
        status_path = Path(arguments.status_path)
    elif arguments.meta_path:
        status_path = Path(arguments.meta_path).with_name("hakari-status.json")
    else:
        status_path = DEFAULT_STATUS_PATH

    samples_path = (
        Path(arguments.samples_path)
        if arguments.samples_path
        else status_path.with_name("memory-samples.jsonl")
    )
    return status_path, samples_path


def command_for(arguments: argparse.Namespace) -> list[str]:
    if arguments.supervise_command:
        return arguments.supervise_command
    if not arguments.script:
        raise SystemExit("script or --supervise-command is required")
    return [sys.executable, arguments.script, *arguments.args]


def stop_child(child: subprocess.Popen[bytes] | None) -> None:
    if child is None or child.poll() is not None:
        return
    terminate_group(child.pid)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def status_record(
    status: str,
    reason: str | None,
    watchdog: MemoryWatchdog,
    child: subprocess.Popen[bytes],
    resume_command: str,
) -> dict:
    record = {
        "status": status,
        "reason": reason,
        "peakOwnedRss": watchdog.peak,
        "lastSample": watchdog.last,
        "childIdentities": [child.pid],
        "checkpointCount": 0,
    }
    if status in {"timed-out", "memory-paused", "interrupted"}:
        record["resumeCommand"] = resume_command
    validate_status(record)
    return record


def run_supervised(arguments: argparse.Namespace, command: list[str]) -> int:
    status_path, samples_path = status_paths(arguments)
    resume_command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "--supervise-command", *command]
    )
    child: subprocess.Popen[bytes] | None = None
    watchdog: MemoryWatchdog | None = None
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(
            f"interrupted by signal {signum}", resume_command=resume_command
        )

    for signum in previous_handlers:
        signal.signal(signum, interrupted)

    try:
        child = subprocess.Popen(command, start_new_session=True)
        watchdog = MemoryWatchdog(
            child.pid,
            ceiling=arguments.memory_ceiling,
            warning=arguments.memory_warning,
            status_path=status_path,
            samples_path=samples_path,
            resume_command=resume_command,
        )
        deadline = time.monotonic() + arguments.timeout_seconds

        while child.poll() is None:
            if watchdog.poll() == "memory-paused":
                raise MemoryPausedError(
                    "memory safety threshold reached",
                    peak_owned_rss=watchdog.peak,
                    last_sample=watchdog.last,
                    child_identities=[child.pid],
                    resume_command=resume_command,
                )
            if time.monotonic() >= deadline:
                raise TimedOutError(
                    f"HAKARI job exceeded {arguments.timeout_seconds:g} seconds",
                    peak_owned_rss=watchdog.peak,
                    last_sample=watchdog.last,
                    child_identities=[child.pid],
                    resume_command=resume_command,
                )
            time.sleep(0.1)

        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)

        atomic_json(
            status_path,
            status_record("completed", None, watchdog, child, resume_command),
        )
        return 0
    except SupervisionError as error:
        stop_child(child)
        if child is not None and watchdog is not None:
            atomic_json(
                status_path,
                status_record(error.status, error.reason, watchdog, child, resume_command),
            )
        return {"memory-paused": 75, "timed-out": 124, "interrupted": 130}[
            error.status
        ]
    except subprocess.CalledProcessError as error:
        if child is not None and watchdog is not None:
            atomic_json(
                status_path,
                status_record("failed", str(error), watchdog, child, resume_command),
            )
        return error.returncode
    finally:
        stop_child(child)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    arguments = parse_arguments()
    command = command_for(arguments)

    if arguments.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if arguments.memory_warning >= arguments.memory_ceiling:
        raise SystemExit("--memory-warning must be lower than --memory-ceiling")
    if not arguments.skip_memory_preflight:
        memory_preflight()

    with BenchmarkLease(
        lease_path=arguments.lease_path,
        meta_path=arguments.meta_path,
        runName="hakari",
        dataset="hakari",
        runtimePath="benchmarks/hakari",
        memoryCeilingBytes=arguments.memory_ceiling,
    ):
        exit_code = run_supervised(arguments, command)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
