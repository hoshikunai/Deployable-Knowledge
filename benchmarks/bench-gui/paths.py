"""Repository locations shared by the bench CLI and GUI.

Importing this module also makes the BEIR harness modules importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCH_ROOT.parent.parent
HARNESS_ROOT = REPOSITORY_ROOT / "benchmarks" / "beir"
RUNS_ROOT = HARNESS_ROOT / "runs"
DATASETS_ROOT = HARNESS_ROOT / "datasets"
RUNTIME_ROOT = REPOSITORY_ROOT / ".cache" / "beir"
STATE_ROOT = REPOSITORY_ROOT / ".cache" / "bench-gui"
JOBS_ROOT = STATE_ROOT / "jobs"
STATIC_ROOT = BENCH_ROOT / "static"
BUILD_ENTRYPOINT = REPOSITORY_ROOT / "build" / "index.js"
SOURCE_ROOT = REPOSITORY_ROOT / "src"

for path in (HARNESS_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
