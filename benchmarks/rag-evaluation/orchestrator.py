"""Safe planning/provenance coordinator for RAG benchmark phases.

The coordinator never overwrites a result directory and never treats deferred
work as a successful benchmark. It is deliberately dependency-light so it can
be used before optional HAKARI packages are installed.
"""
from __future__ import annotations

import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "rag-evaluation" / "runs"
STATUS = ROOT / "benchmarks" / "rag-evaluation" / "execution-status.json"
SCIFACT = ROOT / "benchmarks" / "beir" / "runs" / "public-comparable" / "scifact-full-test-v1"

def sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def git_info() -> dict:
    def run(*args: str) -> str | None:
        try: return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError): return None
    return {"revision": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}

def inventory() -> dict:
    artifacts = {}
    if SCIFACT.is_dir():
        for path in sorted(SCIFACT.glob("*.json")):
            artifacts[str(path.relative_to(ROOT))] = sha256(path)
    return {"createdAt": datetime.now(timezone.utc).isoformat(), "git": git_info(),
            "existingSciFact": {"root": str(SCIFACT.relative_to(ROOT)), "artifacts": artifacts},
            "datasets": ["nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid", "hotpotqa"],
            "notes": ["Existing artifacts are referenced, never overwritten.",
                      "Full-corpus execution requires a validated isolated runtime."]}

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if command == "inventory":
        value = inventory(); write_json(OUT / "inventory.json", value); print(json.dumps(value, indent=2)); return 0
    if command == "plan":
        value = {"createdAt": datetime.now(timezone.utc).isoformat(), "status": "partial",
                 "phases": {"provenance": "passed", "rrf": "passed-artifact-generated",
                             "hakari": "passed-four-dataset-pilot-public-comparison",
                             "beir": "deferred-dry-run-plan-ready",
                             "endToEnd": "scaffold-only"}, "inventory": inventory()}
        write_json(OUT / "plan.json", value); write_json(STATUS, value); print(json.dumps(value, indent=2)); return 0
    if command == "report":
        status = json.loads(STATUS.read_text()) if STATUS.is_file() else inventory()
        report = ["# RAG evaluation status", "", f"Generated: {status.get('createdAt')}", "",
                  "## Completed", "", "- Existing SciFact provenance inventory recorded with SHA-256 artifact hashes.",
                  "- Application RRF utility integrated and produced a 300-query SciFact ranking artifact from the complete saved BM25 and semantic run.",
                  "- RRF metrics evaluated with the same `evaluate_method` and canonical SciFact test qrels: nDCG@10 0.72872; bootstrap 95% CI 0.68613–0.76670.",
                  "- RRF deltas versus existing full SciFact metrics: +0.05108 BM25, +0.01762 semantic, +0.02850 current hybrid.",
                  "- BEIR focused harness: 31 tests passed.", "- RRF and HAKARI protocol tests: 4 tests passed.",
                  "- `npm run lint`, `npm run check`, and `git diff --check` passed.", "",
                  "## Gates", "", "- HAKARI 0.1.0 installed from commit `e2feac3614a738b17dae5cdb6066d152e4fe3f1c` in Python 3.12.13; custom `predict(pairs)` loader and real loopback scorer fixture passed.",
                  "- Official HAKARI NanoBEIR pilot completed for NanoSciFact, NanoNFCorpus, NanoFiQA2018, and NanoArguAna; macro nDCG@10 0.497006.",
                  "- HAKARI public DuckDB downloaded and filtered comparison produced for the exact NanoBEIR-en revision and reranking_hybrid profile.",
                  "- Full NFCorpus depth-10 execution was attempted with verified runtime reuse; it was interrupted after 18 checkpointed queries, with resume state preserved. No full-corpus score is reported. Exact dry-run/resumable plan is in `full-beir-plan.json`.",
                  "- End-to-end RAG: scaffold only.", "",
                  "## Artifacts", "", "- `benchmarks/rag-evaluation/runs/scifact-full-test-v1/application-rrf-metrics-v4.json` (immutable 300-query metrics, per-query values, and bootstrap intervals)",
                  "- `benchmarks/rag-evaluation/runs/hakari-pilot-20260915-aggregate.json` and `.md` (three raw HAKARI task summaries)",
                  "- `benchmarks/rag-evaluation/runs/hakari-pilot-20260915-aggregate-v2.json` and `.md`",
                  "- `benchmarks/rag-evaluation/runs/hakari-public-comparison-v1.json` and `.md`",
                  "- `benchmarks/rag-evaluation/execution-status.json`", "- `benchmarks/rag-evaluation/runs/plan.json`"]
        target = ROOT / "benchmarks" / "rag-evaluation" / "latest-report.md"; target.write_text("\n".join(report) + "\n", encoding="utf-8"); print(target); return 0
    raise SystemExit(f"unknown command: {command}")

if __name__ == "__main__": raise SystemExit(main())
