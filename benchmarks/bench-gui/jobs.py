"""Benchmark job model: settings, run naming, persistence, and the session loop."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import paths
import corpora

# Keys mirror the reranker IDs and token limits in src/lib/server/rag/search/cross-encoders/.
# The app server rejects unknown IDs and over-long limits at startup.
PIPELINES: dict[str, dict[str, Any]] = {
    "rrf": {"label": "RRF only", "reranker": None, "maxTokens": None},
    "rrf-ettin": {"label": "RRF + Ettin 32M", "reranker": "ettin-32m", "maxTokens": 7999},
    "rrf-msmarco": {"label": "RRF + MS MARCO MiniLM", "reranker": "ms-marco-minilm-l6-v2",
                    "maxTokens": 512},
}
DEFAULT_RERANK_MAX_TOKENS = 512
# Experimental candidate dedup modes (src/lib/server/rag/search/candidate-dedup.ts)
DEDUP_MODES = {"none": None, "exact": "dx", "near": "dn", "document-cap": "dc"}
FUSION_LEVELS = ("chunk", "document")
DOCUMENT_ORDERS = ("none", "sum-top2")
SWITCHES = ("off", "on")
SPLITS = ("test", "train", "dev")
PREPARE_TIMEOUT_HOURS = 12

# The harness requests documentDepth * CHUNK_OVERFETCH_FACTOR chunks per query.
CHUNK_OVERFETCH_FACTOR = 2
GIGABYTE = 1024**3

RESUMABLE_STATES = ("memory-paused", "failed", "cancelled", "interrupted", "sessions-exhausted")


@dataclass
class Settings:
    retrievalMultiplier: int = 5
    rerankMultiplier: int = 2
    rrfRankConstant: int = 60
    documentDepth: int = 10
    timeoutMinutes: int = 30
    maxSessions: int = 8
    appPort: int = 4179
    guiPort: int = 8765
    memoryCeilingGb: float = 7.0
    # 0 keeps the default 1,200-character chunking; otherwise a token-chunked corpus copy
    chunkMaxTokens: int = 0
    chunkOverlapTokens: int = 0
    rerankMaxTokens: int = DEFAULT_RERANK_MAX_TOKENS
    embeddingModel: str = corpora.DEFAULT_EMBEDDING_MODEL
    dedupMode: str = "none"
    maxChunksPerDocument: int = 2
    rrfSemanticWeight: float = 1.0
    rrfBm25Weight: float = 1.0
    fusionLevel: str = "chunk"
    documentOrder: str = "none"
    rerankTitle: str = "on"
    # Query split to evaluate: tune on train/dev, report on test
    split: str = "test"

    def chunking(self) -> dict[str, int] | None:
        if self.chunkMaxTokens == 0:
            return None
        return {"maxTokens": self.chunkMaxTokens, "overlapTokens": self.chunkOverlapTokens}

    def validate(self) -> None:
        for name in ("retrievalMultiplier", "rerankMultiplier", "rrfRankConstant",
                     "timeoutMinutes", "maxSessions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 10 <= self.documentDepth <= 100:
            raise ValueError("documentDepth must be between 10 and 100")
        for name in ("appPort", "guiPort"):
            if not 1024 <= getattr(self, name) <= 65535:
                raise ValueError(f"{name} must be between 1024 and 65535")
        if not 1 <= self.memoryCeilingGb <= 64:
            raise ValueError("memoryCeilingGb must be between 1 and 64")
        if self.chunkMaxTokens == 0:
            if self.chunkOverlapTokens != 0:
                raise ValueError("chunkOverlapTokens needs a token chunk size")
        elif not 16 <= self.chunkMaxTokens <= 4096:
            raise ValueError("chunkMaxTokens must be 0 (default) or between 16 and 4096")
        elif not 0 <= self.chunkOverlapTokens < self.chunkMaxTokens:
            raise ValueError("chunkOverlapTokens must be at least 0 and below chunkMaxTokens")
        if not 16 <= self.rerankMaxTokens <= 7999:
            raise ValueError("rerankMaxTokens must be between 16 and 7999")
        if min(self.rrfSemanticWeight, self.rrfBm25Weight) < 0:
            raise ValueError("RRF weights must not be negative")
        if self.rrfSemanticWeight + self.rrfBm25Weight == 0:
            raise ValueError("RRF weights cannot both be 0")
        for name, allowed in (("fusionLevel", FUSION_LEVELS), ("documentOrder", DOCUMENT_ORDERS),
                              ("rerankTitle", SWITCHES), ("split", SPLITS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}")
        if self.dedupMode not in DEDUP_MODES:
            raise ValueError(f"dedupMode must be one of {sorted(DEDUP_MODES)}")
        if self.maxChunksPerDocument < 1:
            raise ValueError("maxChunksPerDocument must be at least 1")
        if self.embeddingModel not in corpora.EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model {self.embeddingModel!r}; "
                             f"choose from {sorted(corpora.EMBEDDING_MODELS)}")
        model_limit = corpora.EMBEDDING_MODELS[self.embeddingModel]["maxTokens"]
        if self.chunkMaxTokens > model_limit:
            raise ValueError(f"{self.embeddingModel} reads at most {model_limit} tokens; "
                             f"lower the chunk size")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Settings":
        known = {item.name: item.type for item in fields(cls)}
        unknown = set(value) - set(known)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        defaults = asdict(cls())
        merged = {**defaults, **value}
        casts = {"memoryCeilingGb": float, "rrfSemanticWeight": float, "rrfBm25Weight": float,
                 "embeddingModel": str, "dedupMode": str, "fusionLevel": str,
                 "documentOrder": str, "rerankTitle": str, "split": str}
        settings = cls(**{name: casts.get(name, int)(merged[name]) for name in known})
        settings.validate()
        return settings


SETTINGS_PATH = paths.STATE_ROOT / "settings.json"


def load_settings() -> Settings:
    if not SETTINGS_PATH.is_file():
        return Settings()
    return Settings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))


def save_settings(settings: Settings) -> None:
    settings.validate()
    write_json(SETTINGS_PATH, asdict(settings))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class JobSpec:
    dataset: str
    pipeline: str
    queries: str  # "all" or a positive integer as text
    seed: int
    settings: Settings
    kind: str = "benchmark"  # or "prepare": build a corpus copy for the chunk settings
    runName: str = ""
    state: str = "queued"
    message: str = ""
    sessions: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        """Run directory name under benchmarks/beir/runs (run_suite prefixes the dataset)."""
        return f"{self.dataset}-{self.runName}"

    @property
    def chunking(self) -> dict[str, int] | None:
        return self.settings.chunking()

    @property
    def embedding_model(self) -> str:
        return self.settings.embeddingModel

    @property
    def output_chunks(self) -> int:
        return self.settings.documentDepth * CHUNK_OVERFETCH_FACTOR

    def pipeline_config(self) -> dict[str, Any]:
        """The HybridPipelineConfig the app server must report for this job."""
        return {
            "reranker": PIPELINES[self.pipeline]["reranker"],
            "retrievalMultiplier": self.settings.retrievalMultiplier,
            "rerankMultiplier": self.settings.rerankMultiplier,
            "rrfRankConstant": self.settings.rrfRankConstant,
            "rerankMaxTokens": self.settings.rerankMaxTokens,
            "dedup": self.settings.dedupMode,
            "maxChunksPerDocument": self.settings.maxChunksPerDocument,
            "rrfSemanticWeight": self.settings.rrfSemanticWeight,
            "rrfBm25Weight": self.settings.rrfBm25Weight,
            "fusionLevel": self.settings.fusionLevel,
            "documentOrder": self.settings.documentOrder,
            "rerankTitle": self.settings.rerankTitle == "on",
        }

    def corpus(self) -> dict[str, Any]:
        """Which prepared corpus copy the job searches; recorded in run-config.json."""
        return {"runtimeId": corpora.runtime_id(self.dataset, self.chunking, self.embedding_model),
                "chunking": self.chunking,
                "embeddingModel": self.embedding_model}

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runId"] = self.run_id
        value["pipelineConfig"] = self.pipeline_config()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobSpec":
        known = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known}
        payload["settings"] = Settings.from_dict(payload["settings"])
        return cls(**payload)


def validate_job(dataset: str, pipeline: str, queries: str, settings: Settings) -> None:
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose from {sorted(PIPELINES)}")
    if queries != "all" and (not queries.isdigit() or int(queries) < 1):
        raise ValueError("queries must be 'all' or a positive integer")
    limit = PIPELINES[pipeline]["maxTokens"]
    if limit is not None and settings.rerankMaxTokens > limit:
        raise ValueError(f"{PIPELINES[pipeline]['label']} reads at most {limit} tokens; "
                         f"lower the reranker max tokens setting")
    if not corpora.has_split(dataset, settings.split):
        raise ValueError(f"{dataset} has no {settings.split} split")
    corpora.require_prepared(dataset, settings.chunking(), settings.embeddingModel)


def make_run_name(spec: JobSpec, now: datetime | None = None) -> str:
    """Describe the exact configuration so run names can't drift from what ran."""
    out = spec.output_chunks
    parts = [spec.pipeline, f"p{out * spec.settings.retrievalMultiplier}"]
    if PIPELINES[spec.pipeline]["reranker"] is not None:
        parts.append(f"s{out * spec.settings.rerankMultiplier}")
    parts.append(f"o{out}")
    if spec.settings.rrfRankConstant != 60:
        parts.append(f"k{spec.settings.rrfRankConstant}")
    reranked = PIPELINES[spec.pipeline]["reranker"] is not None
    if reranked and spec.settings.rerankMaxTokens != DEFAULT_RERANK_MAX_TOKENS:
        parts.append(f"r{spec.settings.rerankMaxTokens}")
    if (spec.settings.rrfSemanticWeight, spec.settings.rrfBm25Weight) != (1, 1):
        parts.append(f"w{spec.settings.rrfSemanticWeight:g}x{spec.settings.rrfBm25Weight:g}")
    if spec.settings.fusionLevel == "document":
        parts.append("fdoc")
    if spec.settings.documentOrder == "sum-top2":
        parts.append("os2")
    if reranked and spec.settings.rerankTitle == "off":
        parts.append("nt")
    dedup_tag = DEDUP_MODES[spec.settings.dedupMode]
    if dedup_tag == "dc":
        parts.append(f"dc{spec.settings.maxChunksPerDocument}")
    elif dedup_tag:
        parts.append(dedup_tag)
    variant = corpora.variant_id(spec.chunking, spec.embedding_model)
    if variant != corpora.DEFAULT_VARIANT:
        parts.append(variant)
    if spec.settings.split != "test":
        parts.append(spec.settings.split)
    parts.append("all" if spec.queries == "all" else f"{spec.queries}q-seed{spec.seed}")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S-%f")
    parts.append(stamp)
    return "-".join(parts)


def create_jobs(datasets: list[str], pipeline: str, queries: str, seeds: list[int],
                settings: Settings) -> list[JobSpec]:
    if not datasets:
        raise ValueError("Select at least one corpus")
    if not seeds:
        raise ValueError("Provide at least one seed")
    settings.validate()
    # A full run evaluates every query, so extra seeds would only repeat it.
    effective_seeds = seeds[:1] if queries == "all" else seeds
    jobs: list[JobSpec] = []
    for dataset in datasets:
        validate_job(dataset, pipeline, queries, settings)
        for seed in effective_seeds:
            spec = JobSpec(dataset=dataset, pipeline=pipeline, queries=queries, seed=seed,
                           settings=settings)
            spec.runName = make_run_name(spec)
            jobs.append(spec)
    return jobs


def create_prepare_jobs(datasets: list[str], settings: Settings) -> list[JobSpec]:
    """One job per dataset that builds a corpus copy for the chunk and embedding settings."""
    settings.validate()
    chunking = settings.chunking()
    variant = corpora.variant_id(chunking, settings.embeddingModel)
    if not datasets:
        raise ValueError("Select at least one corpus")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    jobs = []
    for dataset in datasets:
        status = corpora.variant_status(dataset, chunking, settings.embeddingModel)
        if status["state"] == "prepared":
            described = corpora.describe_variant(chunking, settings.embeddingModel)
            raise ValueError(f"{dataset} is already prepared with {described}")
        jobs.append(JobSpec(dataset=dataset, pipeline="rrf", queries="all", seed=0,
                            settings=settings, kind="prepare",
                            runName=f"prepare-{variant}-{stamp}"))
    return jobs


def job_path(run_id: str) -> Path:
    return paths.JOBS_ROOT / f"{run_id}.json"


def save_job(spec: JobSpec) -> None:
    write_json(job_path(spec.run_id), spec.to_dict())


def load_job(run_id: str) -> JobSpec:
    path = job_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No bench job for {run_id}. Only runs started with 'npm run bench' can be resumed here."
        )
    return JobSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_jobs() -> list[JobSpec]:
    if not paths.JOBS_ROOT.is_dir():
        return []
    jobs = [JobSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(paths.JOBS_ROOT.glob("*.json"))]
    return jobs


def run_directory(spec: JobSpec) -> Path:
    return paths.RUNS_ROOT / spec.run_id


def completed_query_count(directory: Path) -> int:
    status_path = directory / "run-status.json"
    if not status_path.is_file():
        return 0
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return len(status.get("completedQueryIds", []))


def suite_arguments(spec: JobSpec, resume: bool) -> argparse.Namespace:
    """The namespace previously hand-written in run-command.txt."""
    ceiling = int(spec.settings.memoryCeilingGb * GIGABYTE)
    if spec.kind == "prepare":
        return argparse.Namespace(
            split="test", query_counts="all", sample_seed=42,
            search_depth=spec.settings.documentDepth, port=spec.settings.appPort,
            startup_timeout=120, no_shared_model_cache=False, run_name=spec.runName,
            resume=False, reuse_existing_schema=False, reuse_existing_runtime=False,
            reuse_document_mapping=None,
            runtime_id=corpora.runtime_id(spec.dataset, spec.chunking, spec.embedding_model),
            timeout_seconds=PREPARE_TIMEOUT_HOURS * 3600, prepare_only=True,
            chunking=spec.chunking, embedding_model=spec.embedding_model,
            memory_ceiling=ceiling, memory_warning=int(ceiling * 0.85),
        )
    mapping = corpora.require_prepared(spec.dataset, spec.chunking, spec.embedding_model)
    return argparse.Namespace(
        split=spec.settings.split,
        query_counts=spec.queries,
        sample_seed=spec.seed,
        search_depth=spec.settings.documentDepth,
        port=spec.settings.appPort,
        startup_timeout=120,
        no_shared_model_cache=False,
        run_name=spec.runName,
        resume=resume,
        reuse_existing_schema=True,
        reuse_existing_runtime=True,
        reuse_document_mapping=mapping.resolve(),
        runtime_id=corpora.runtime_id(spec.dataset, spec.chunking, spec.embedding_model),
        timeout_seconds=spec.settings.timeoutMinutes * 60,
        pipeline=spec.pipeline_config(),
        corpus=spec.corpus(),
        embedding_model=spec.embedding_model,
        memory_ceiling=ceiling,
        memory_warning=int(ceiling * 0.85),
    )


def git_output(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=paths.REPOSITORY_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def build_status() -> dict[str, Any]:
    build_time = paths.BUILD_ENTRYPOINT.stat().st_mtime if paths.BUILD_ENTRYPOINT.is_file() else None
    newest_source = max((path.stat().st_mtime for path in paths.SOURCE_ROOT.rglob("*")
                         if path.is_file()), default=0.0)
    return {
        "buildTime": build_time,
        "newestSourceTime": newest_source,
        "stale": build_time is None or newest_source > build_time,
        "sourceCommit": git_output("rev-parse", "HEAD"),
        "sourceDirty": bool(git_output("status", "--porcelain", "--", "src")),
    }


def record_execution_meta(spec: JobSpec) -> None:
    directory = run_directory(spec)
    if not directory.is_dir():
        return
    write_json(directory / "execution-meta.json", {
        **build_status(),
        "pipeline": spec.pipeline,
        "pipelineConfig": spec.pipeline_config(),
        "corpus": spec.corpus(),
        "settings": asdict(spec.settings),
        "sessions": spec.history,
    })


Log = Callable[[str], None]


def record_session(spec: JobSpec, outcome: str, before: int, after: int) -> None:
    spec.history.append({"session": spec.sessions, "outcome": outcome,
                         "completedBefore": before, "completedAfter": after,
                         "at": datetime.now(timezone.utc).isoformat()})
    record_execution_meta(spec)


def print_flushed(text: str) -> None:
    print(text, flush=True)


def execute(spec: JobSpec, log: Log = print_flushed) -> str:
    """Run a job to completion, resuming after session timeouts. Returns the final state."""
    import run_suite
    from benchmarks.benchmark_lease import MemoryPausedError, TimedOutError

    if build_status()["buildTime"] is None:
        raise RuntimeError("The app build is missing; run 'npm run build:electron' first")
    if spec.kind == "prepare":
        return execute_prepare(spec, log)
    directory = run_directory(spec)
    spec.state = "running"
    spec.message = ""
    save_job(spec)

    for _session in range(spec.settings.maxSessions):
        resume = (directory / "run-config.json").is_file()
        before = completed_query_count(directory)
        spec.sessions += 1
        log(f"{spec.run_id}: session {spec.sessions} ({'resume' if resume else 'start'}, "
            f"{before} queries done)")
        outcome = "complete"
        try:
            run_suite.run_dataset(spec.dataset, suite_arguments(spec, resume), spec.runName)
        except TimedOutError:
            outcome = "timed-out"
        except MemoryPausedError:
            outcome = "memory-paused"
        except KeyboardInterrupt:
            record_session(spec, "cancelled", before, completed_query_count(directory))
            raise
        record_session(spec, outcome, before, completed_query_count(directory))
        after = completed_query_count(directory)

        if outcome == "complete":
            spec.state = "complete"
            save_job(spec)
            return spec.state
        if outcome == "memory-paused":
            spec.state = "memory-paused"
            spec.message = "Stopped at the memory ceiling; free memory, then resume."
            save_job(spec)
            return spec.state
        if after <= before:
            spec.state = "failed"
            spec.message = "Session timed out without completing any queries."
            save_job(spec)
            return spec.state
        log(f"{spec.run_id}: session timed out after {after} queries; resuming")
        save_job(spec)

    spec.state = "sessions-exhausted"
    spec.message = f"Used all {spec.settings.maxSessions} sessions; resume to continue."
    save_job(spec)
    return spec.state


def execute_prepare(spec: JobSpec, log: Log) -> str:
    """Ingest the whole corpus into a new copy with the job's chunk settings.

    Ingestion is not resumable: an interrupted copy stays marked "preparing" and the next
    prepare job for the same settings rebuilds it from scratch.
    """
    import shutil

    import run_suite

    chunking = spec.chunking
    variant = corpora.describe_variant(chunking, spec.embedding_model)
    runtime = corpora.runtime_id(spec.dataset, chunking, spec.embedding_model)
    corpora.start_preparing(spec.dataset, chunking, spec.embedding_model)
    spec.state = "running"
    spec.message = ""
    spec.sessions += 1
    save_job(spec)
    log(f"{spec.dataset}: preparing {variant} ({runtime}); this ingests the whole corpus")

    run_suite.run_dataset(spec.dataset, suite_arguments(spec, resume=False), spec.runName)

    directory = run_directory(spec)
    marker = corpora.finish_preparing(spec.dataset, chunking, spec.embedding_model,
                                      directory / corpora.MAPPING_FILE)
    # The ingestion run folder only carried the mapping, which now lives beside the database.
    shutil.rmtree(directory)
    supervisor = paths.RUNS_ROOT / f".supervisor-{spec.run_id}"
    if supervisor.is_dir():
        shutil.rmtree(supervisor)
    spec.state = "complete"
    spec.message = (f"Prepared {marker['documentCount']} documents as "
                    f"{marker['chunkCount']} chunks")
    save_job(spec)
    log(f"{spec.dataset}: {spec.message}")
    return spec.state
