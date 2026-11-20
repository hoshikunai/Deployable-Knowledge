"""Bring a benchmark-only copy of the repository up to date before a job runs.

The copy follows one branch on origin and only ever fast-forwards. Local edits, another
checked-out branch, or history that has diverged stop the sync instead of being discarded.
Dependencies are reinstalled only when package-lock.json changed, and the app is rebuilt only
when the build is older than the source.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import paths
from benchmarks.process_group import new_group_kwargs
from jobs import build_status

Log = Callable[[str], None]
GUI_PREFIX = "benchmarks/bench-gui/"


class SyncError(RuntimeError):
    pass


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise SyncError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def run_logged(command: list[str], root: Path, log: Log) -> None:
    """Stream a long command's output to the job log; a failure stops the sync."""
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               **new_group_kwargs())
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip("\n"))
    if process.wait() != 0:
        raise SyncError(f"{' '.join(command)} exited with {process.returncode}")


def npm(*arguments: str) -> list[str]:
    return [shutil.which("npm") or "npm", *arguments]


def sync_repository(branch: str, log: Log, root: Path = paths.REPOSITORY_ROOT,
                    build_stale: Callable[[], bool] | None = None) -> bool:
    """Fast-forward to origin/<branch>, reinstall and rebuild as needed.

    Returns True when the bench GUI's own files changed, so the running GUI should restart.
    """
    local_changes = git(root, "status", "--porcelain", "--untracked-files=no")
    if local_changes:
        files = ", ".join(line.split(maxsplit=1)[1] for line in local_changes.splitlines()[:5])
        raise SyncError(f"the copy has local changes ({files}); commit or revert them on this machine")
    checked_out = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if checked_out != branch:
        raise SyncError(f"the copy has {checked_out} checked out, but syncs {branch}")

    log(f"Sync: fetching origin/{branch}")
    git(root, "fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    old = git(root, "rev-parse", "HEAD")
    new = git(root, "rev-parse", f"origin/{branch}")
    changed: list[str] = []
    if old == new:
        log(f"Sync: already at {new[:8]}")
    else:
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", old, new], cwd=root)
        if ancestor.returncode != 0:
            raise SyncError(f"local {branch} has commits that origin/{branch} lacks; "
                            "it can't be fast-forwarded")
        git(root, "merge", "--ff-only", new)
        changed = git(root, "diff", "--name-only", old, new).splitlines()
        log(f"Sync: fast-forwarded {old[:8]} → {new[:8]} ({len(changed)} files)")

    if "package-lock.json" in changed:
        log("Sync: package-lock.json changed; running npm ci")
        run_logged(npm("ci"), root, log)
    stale = build_stale() if build_stale else build_status()["stale"]
    if stale:
        log("Sync: the build is older than the source; running npm run build:electron")
        run_logged(npm("run", "build:electron"), root, log)

    gui_changed = any(path.startswith(GUI_PREFIX) for path in changed)
    if gui_changed:
        log("Sync: the bench GUI changed; restart it to load the new version "
            "(this job already uses the new harness)")
    return gui_changed
