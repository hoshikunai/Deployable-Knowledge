"""Local web GUI for the BEIR bench. Runs queued jobs one at a time."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import paths
from corpora import list_corpora
from jobs import (
    PIPELINES,
    RESUMABLE_STATES,
    Settings,
    build_status,
    completed_query_count,
    create_jobs,
    create_prepare_jobs,
    job_path,
    list_jobs,
    load_job,
    load_settings,
    run_directory,
    save_job,
    save_settings,
)
from runs import delete_run, list_runs, run_path
from scores import build_scores

BUILD_ITEM = "__build__"
INGEST_PROGRESS = re.compile(r"Ingested (\d+)/(\d+) documents")
LOG_LIMIT = 5000
CANCEL_GRACE_SECONDS = 60


class Runner:
    """A single worker thread that runs queued jobs in child processes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.queue: deque[str] = deque()
        self.current: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.log_lines: deque[dict[str, Any]] = deque(maxlen=LOG_LIMIT)
        self.sequence = 0
        # Latest "Ingested n/total" of the running prepare job
        self.ingest_progress: tuple[int, int] | None = None
        self.mark_orphaned_jobs()
        threading.Thread(target=self.work, daemon=True).start()

    def mark_orphaned_jobs(self) -> None:
        for spec in list_jobs():
            if spec.state in ("queued", "running"):
                spec.state = "interrupted"
                spec.message = "The GUI stopped before this job finished; resume to continue."
                save_job(spec)

    def log(self, text: str) -> None:
        print(text, flush=True)
        progress = INGEST_PROGRESS.search(text)
        with self.lock:
            if progress:
                self.ingest_progress = (int(progress.group(1)), int(progress.group(2)))
            self.log_lines.append({"seq": self.sequence, "time": time.time(), "text": text})
            self.sequence += 1

    def since(self, sequence: int) -> list[dict[str, Any]]:
        with self.lock:
            return [line for line in self.log_lines if line["seq"] >= sequence]

    def enqueue(self, item: str) -> None:
        with self.lock:
            if item == self.current or item in self.queue:
                raise HTTPException(409, f"{item} is already queued or running")
            self.queue.append(item)
        self.wakeup.set()

    def busy_with(self, run_id: str) -> bool:
        with self.lock:
            return run_id == self.current or run_id in self.queue

    def remove_queued(self, run_id: str) -> None:
        with self.lock:
            if run_id not in self.queue:
                raise HTTPException(404, f"{run_id} is not queued")
            self.queue.remove(run_id)
        spec = load_job(run_id)
        if not run_directory(spec).is_dir():
            # Nothing ran yet, so there is nothing to resume or keep.
            job_path(run_id).unlink()
            return
        spec.state = "cancelled"
        spec.message = "Removed from the queue; resume to continue."
        save_job(spec)

    def cancel_current(self) -> None:
        with self.lock:
            process = self.process
        if process is None or process.poll() is not None:
            raise HTTPException(409, "Nothing is running")
        self.log("Cancelling: the app server and search process are being stopped…")
        os.killpg(process.pid, signal.SIGINT)
        threading.Thread(target=self.force_stop, args=(process,), daemon=True).start()

    def force_stop(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=CANCEL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self.log("Cancel grace period passed; terminating.")
            os.killpg(process.pid, signal.SIGTERM)

    def work(self) -> None:
        while True:
            self.wakeup.wait()
            with self.lock:
                if not self.queue:
                    self.wakeup.clear()
                    continue
                self.current = self.queue.popleft()
                self.ingest_progress = None
            try:
                self.run_item(self.current)
            except Exception as error:  # keep the worker alive for the next job
                self.log(f"{self.current}: {type(error).__name__}: {error}")
            finally:
                with self.lock:
                    self.current = None
                    self.process = None

    def run_item(self, item: str) -> None:
        if item == BUILD_ITEM:
            self.log("Building the app (npm run build:electron)…")
            command = ["npm", "run", "build:electron"]
        else:
            self.log(f"Starting {item}")
            command = [sys.executable, "-u", str(paths.BENCH_ROOT / "bench.py"), "run-one", item]
        process = subprocess.Popen(
            command, cwd=paths.REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        with self.lock:
            self.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.log(line.rstrip("\n"))
        process.wait()
        self.log(f"{'Build' if item == BUILD_ITEM else item} finished (exit {process.returncode})")

    def stop_on_exit(self) -> None:
        """Closing the GUI stops the running job so no app server is left behind."""
        with self.lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        print("Stopping the running job before exit…", flush=True)
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=CANCEL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"current": self.current, "queue": list(self.queue)}


runner: Runner | None = None
app = FastAPI(title="BEIR bench")


def get_runner() -> Runner:
    if runner is None:
        raise RuntimeError("Runner not started")
    return runner


def job_progress(run_id: str) -> dict[str, Any]:
    spec = load_job(run_id)
    if spec.kind == "prepare":
        progress = get_runner().ingest_progress if get_runner().current == run_id else None
        completed, total = progress or (0, None)
        return {**spec.to_dict(), "completed": completed, "total": total}
    directory = run_directory(spec)
    config_path = directory / "run-config.json"
    total = None
    if config_path.is_file():
        total = len(json.loads(config_path.read_text(encoding="utf-8")).get("queryIds", []))
    elif spec.queries != "all":
        total = int(spec.queries)
    return {**spec.to_dict(), "completed": completed_query_count(directory), "total": total}


class JobRequest(BaseModel):
    datasets: list[str]
    pipeline: str
    queries: str
    seeds: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(paths.STATIC_ROOT / "index.html")


@app.get("/api/options")
def options() -> dict[str, Any]:
    settings = load_settings()
    return {
        "corpora": list_corpora(settings.chunking()),
        "pipelines": [{"key": key, **value} for key, value in PIPELINES.items()],
        "settings": asdict(settings),
        "defaults": asdict(Settings()),
    }


@app.put("/api/settings")
def update_settings(value: dict[str, Any]) -> dict[str, Any]:
    try:
        settings = Settings.from_dict(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    save_settings(settings)
    return asdict(settings)


@app.get("/api/build")
def build() -> dict[str, Any]:
    return build_status()


@app.post("/api/build")
def rebuild() -> dict[str, Any]:
    get_runner().enqueue(BUILD_ITEM)
    return get_runner().snapshot()


@app.get("/api/state")
def state() -> dict[str, Any]:
    snapshot = get_runner().snapshot()
    current = snapshot["current"]
    return {
        "current": job_progress(current) if current and current != BUILD_ITEM else current,
        "queue": [job_progress(item) if item != BUILD_ITEM else {"runId": item}
                  for item in snapshot["queue"]],
    }


@app.post("/api/jobs")
def start_jobs(request: JobRequest) -> dict[str, Any]:
    seeds = [value.strip() for value in request.seeds.split(",") if value.strip()]
    try:
        jobs = create_jobs(request.datasets, request.pipeline, request.queries.strip().lower(),
                           [int(seed) for seed in seeds], load_settings())
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    for spec in jobs:
        save_job(spec)
        get_runner().enqueue(spec.run_id)
    return {"queued": [spec.run_id for spec in jobs]}


@app.get("/api/corpora")
def corpora_for(chunkMaxTokens: int = 0, chunkOverlapTokens: int = 0) -> list[dict[str, Any]]:
    """Corpus copy status for chunk values that may not be saved yet."""
    try:
        settings = Settings.from_dict({"chunkMaxTokens": chunkMaxTokens,
                                       "chunkOverlapTokens": chunkOverlapTokens})
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return list_corpora(settings.chunking())


class PrepareRequest(BaseModel):
    datasets: list[str]


@app.post("/api/corpora/prepare")
def prepare_corpora(request: PrepareRequest) -> dict[str, Any]:
    try:
        jobs = create_prepare_jobs(request.datasets, load_settings())
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    for spec in jobs:
        save_job(spec)
        get_runner().enqueue(spec.run_id)
    return {"queued": [spec.run_id for spec in jobs]}


@app.post("/api/jobs/cancel")
def cancel_job() -> dict[str, Any]:
    get_runner().cancel_current()
    return get_runner().snapshot()


@app.delete("/api/jobs/{run_id}")
def remove_job(run_id: str) -> dict[str, Any]:
    get_runner().remove_queued(run_id)
    return get_runner().snapshot()


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    return list_runs()


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict[str, Any]:
    try:
        spec = load_job(run_id)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    if spec.state not in RESUMABLE_STATES:
        raise HTTPException(409, f"{run_id} is {spec.state}; nothing to resume")
    spec.state = "queued"
    spec.message = ""
    save_job(spec)
    get_runner().enqueue(run_id)
    return get_runner().snapshot()


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str) -> dict[str, Any]:
    if get_runner().busy_with(run_id):
        raise HTTPException(409, "Cancel or dequeue this run before deleting it")
    try:
        removed = delete_run(run_id)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    get_runner().log(f"Deleted {', '.join(removed)}")
    return {"removed": removed}


@app.get("/api/scores")
def scores(runs: str) -> dict[str, Any]:
    run_ids = [value for value in runs.split(",") if value]
    try:
        for run_id in run_ids:
            run_path(run_id)
        result = build_scores(run_ids)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise HTTPException(400, str(error)) from error
    # The raw JSON also goes to the terminal, as requested for every Scores view.
    print("===== SCORES (raw JSON) =====", flush=True)
    print(json.dumps(result["raw"], indent=2), flush=True)
    return result


@app.get("/api/events")
async def events(after: int = 0) -> EventSourceResponse:
    async def stream():
        sequence = after
        while True:
            for line in get_runner().since(sequence):
                sequence = line["seq"] + 1
                yield {"id": str(line["seq"]), "data": json.dumps(line)}
            await asyncio.sleep(0.3)

    return EventSourceResponse(stream())


app.mount("/static", StaticFiles(directory=paths.STATIC_ROOT), name="static")


def open_in_browser(url: str) -> None:
    if shutil.which("wslview"):
        subprocess.Popen(["wslview", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    webbrowser.open(url)


def serve(port: int, open_browser: bool) -> None:
    global runner
    runner = Runner()
    atexit.register(runner.stop_on_exit)
    url = f"http://127.0.0.1:{port}"
    print(f"BEIR bench GUI: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(1.0, open_in_browser, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
