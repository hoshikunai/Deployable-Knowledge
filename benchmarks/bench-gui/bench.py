"""Deployable Knowledge BEIR bench.

    npm run bench                                   open the GUI
    npm run bench -- run scifact arguana --pipeline rrf-ettin --queries 100 --seeds 42,314
    npm run bench -- prepare nfcorpus --chunk-tokens 256 --overlap-tokens 32
    npm run bench -- resume <run-id>
    npm run bench -- list
    npm run bench -- scores <run-id> [<run-id> ...] [--markdown]
    npm run bench -- delete <run-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import paths  # noqa: F401  (puts the BEIR harness on sys.path)
from jobs import (
    PIPELINES,
    JobSpec,
    Settings,
    create_jobs,
    create_prepare_jobs,
    execute,
    load_job,
    load_settings,
    save_job,
)
from runs import delete_run, list_runs
from scores import build_scores

RAW_SCORES_BANNER = "===== SCORES (raw JSON) ====="


def parse_seeds(text: str) -> list[int]:
    try:
        return [int(value) for value in text.replace(" ", "").split(",") if value]
    except ValueError as error:
        raise SystemExit(f"Seeds must be comma-separated integers: {text!r}") from error


def print_raw_scores(run_ids: list[str]) -> dict:
    result = build_scores(run_ids)
    print(RAW_SCORES_BANNER, flush=True)
    print(json.dumps(result["raw"], indent=2), flush=True)
    return result


def run_job(spec: JobSpec) -> str:
    """Execute one job, recording failures on the job so the GUI can show them."""
    try:
        state = execute(spec)
    except KeyboardInterrupt:
        spec.state = "cancelled"
        spec.message = "Cancelled; resume to continue from the last saved query."
        save_job(spec)
        print(f"{spec.run_id}: cancelled", flush=True)
        raise
    except Exception as error:  # surface every failure on the job record
        spec.state = "failed"
        spec.message = f"{type(error).__name__}: {error}"
        save_job(spec)
        print(f"{spec.run_id}: failed — {spec.message}", flush=True)
        return spec.state

    if state == "complete" and spec.kind == "prepare":
        print(f"{spec.run_id}: {spec.message}", flush=True)
    elif state == "complete":
        print(f"{spec.run_id}: complete", flush=True)
        print_raw_scores([spec.run_id])
    else:
        print(f"{spec.run_id}: {state} — {spec.message}\n"
              f"Resume with: npm run bench -- resume {spec.run_id}", flush=True)
    return state


def settings_with_overrides(arguments: argparse.Namespace) -> Settings:
    overrides = {
        "retrievalMultiplier": arguments.retrieval_multiplier,
        "rerankMultiplier": arguments.rerank_multiplier,
        "rrfRankConstant": arguments.rrf_k,
        "documentDepth": arguments.depth,
        "timeoutMinutes": arguments.timeout_minutes,
        "maxSessions": arguments.max_sessions,
        "chunkMaxTokens": arguments.chunk_tokens,
        "chunkOverlapTokens": arguments.overlap_tokens,
        "rerankMaxTokens": getattr(arguments, "rerank_max_tokens", None),
    }
    base = asdict(load_settings())
    base.update({key: value for key, value in overrides.items() if value is not None})
    return Settings.from_dict(base)


def command_run(arguments: argparse.Namespace) -> int:
    queries = arguments.queries.lower()
    jobs = create_jobs(arguments.datasets, arguments.pipeline, queries,
                       parse_seeds(arguments.seeds), settings_with_overrides(arguments))
    for spec in jobs:
        save_job(spec)
    print("Queued: " + ", ".join(spec.run_id for spec in jobs), flush=True)
    states = [run_job(spec) for spec in jobs]
    return 0 if all(state == "complete" for state in states) else 1


def command_prepare(arguments: argparse.Namespace) -> int:
    jobs = create_prepare_jobs(arguments.datasets, settings_with_overrides(arguments))
    for spec in jobs:
        save_job(spec)
    states = [run_job(spec) for spec in jobs]
    return 0 if all(state == "complete" for state in states) else 1


def command_resume(arguments: argparse.Namespace) -> int:
    spec = load_job(arguments.run_id)
    if spec.state == "complete":
        print(f"{spec.run_id} is already complete.")
        print_raw_scores([spec.run_id])
        return 0
    return 0 if run_job(spec) == "complete" else 1


def command_list(_arguments: argparse.Namespace) -> int:
    for run in list_runs():
        ndcg = run["ndcg10"].get("hybrid")
        score = f"{ndcg:.5f}" if ndcg is not None else "   –   "
        progress = f"{run['completed']}/{run['queryCount']}"
        print(f"{score}  {run['status']:<10} {progress:>11}  {run['id']}\n"
              f"{'':>9}{run['pipelineLabel']}")
    return 0


def command_scores(arguments: argparse.Namespace) -> int:
    result = print_raw_scores(arguments.run_ids)
    if arguments.markdown:
        print(result["markdown"])
    return 0


def command_delete(arguments: argparse.Namespace) -> int:
    if not arguments.yes:
        answer = input(f"Permanently delete {arguments.run_id}? [y/N] ").strip().lower()
        if answer != "y":
            print("Not deleted.")
            return 1
    for path in delete_run(arguments.run_id):
        print(f"Deleted {path}")
    return 0


def command_gui(arguments: argparse.Namespace) -> int:
    from server import serve

    serve(arguments.port or load_settings().guiPort, open_browser=not arguments.no_browser)
    return 0


def command_run_one(arguments: argparse.Namespace) -> int:
    """Internal: the GUI runs each queued job in its own process through this command."""
    return 0 if run_job(load_job(arguments.run_id)) == "complete" else 1


def add_chunk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-tokens", type=int,
                        help="Chunk size in tokens (0 = default 1,200 characters).")
    parser.add_argument("--overlap-tokens", type=int, help="Chunk overlap in tokens.")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="npm run bench --", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")

    gui = commands.add_parser("gui", help="Open the benchmark GUI (default).")
    gui.add_argument("--port", type=int)
    gui.add_argument("--no-browser", action="store_true")
    gui.set_defaults(handler=command_gui)

    run = commands.add_parser("run", help="Run benchmarks in this terminal.")
    run.add_argument("datasets", nargs="+")
    run.add_argument("--pipeline", choices=sorted(PIPELINES), default="rrf")
    run.add_argument("--queries", default="all", help="'all' or a sample size such as 100.")
    run.add_argument("--seeds", default="42", help="Comma-separated sampling seeds.")
    run.add_argument("--retrieval-multiplier", type=int)
    run.add_argument("--rerank-multiplier", type=int)
    run.add_argument("--rrf-k", type=int)
    run.add_argument("--depth", type=int, help="Documents evaluated per query (default 10).")
    run.add_argument("--timeout-minutes", type=int)
    run.add_argument("--max-sessions", type=int)
    run.add_argument("--rerank-max-tokens", type=int)
    add_chunk_options(run)
    run.set_defaults(handler=command_run)

    prepare = commands.add_parser("prepare", help="Build token-chunked copies of corpora.")
    prepare.add_argument("datasets", nargs="+")
    add_chunk_options(prepare)
    for name in ("retrieval_multiplier", "rerank_multiplier", "rrf_k", "depth",
                 "timeout_minutes", "max_sessions"):
        prepare.set_defaults(**{name: None})
    prepare.set_defaults(handler=command_prepare)

    resume = commands.add_parser("resume", help="Continue an unfinished run.")
    resume.add_argument("run_id")
    resume.set_defaults(handler=command_resume)

    listing = commands.add_parser("list", help="List runs, newest first.")
    listing.set_defaults(handler=command_list)

    score = commands.add_parser("scores", help="Print scores as raw JSON.")
    score.add_argument("run_ids", nargs="+")
    score.add_argument("--markdown", action="store_true", help="Also print the Markdown tables.")
    score.set_defaults(handler=command_scores)

    delete = commands.add_parser("delete", help="Delete a run and its supervisor files.")
    delete.add_argument("run_id")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(handler=command_delete)

    run_one = commands.add_parser("run-one")
    run_one.add_argument("run_id")
    run_one.set_defaults(handler=command_run_one)

    arguments = parser.parse_args(argv)
    if arguments.command is None:
        arguments = parser.parse_args(["gui", *argv])
    return arguments


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    try:
        return arguments.handler(arguments)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
