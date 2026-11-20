"""Sync tests against real git repositories: a bare origin and a benchmark-only clone."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import paths  # noqa: F401  (puts the BEIR harness on sys.path)
import jobs
import sync

BRANCH = "docling-beir"
IDENTITY = ["-c", "user.name=Bench Test", "-c", "user.email=bench@example.invalid"]


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *IDENTITY, *arguments], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(root: Path, files: dict[str, str], message: str) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", message)


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.origin = base / "origin.git"
        self.author = base / "author"
        self.copy = base / "copy"
        subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], check=True)
        subprocess.run(["git", "init", "-q", "-b", BRANCH, str(self.author)], check=True)
        commit(self.author, {"package-lock.json": "{}", "src/app.ts": "1",
                             "benchmarks/bench-gui/server.py": "1"}, "start")
        git(self.author, "remote", "add", "origin", str(self.origin))
        git(self.author, "push", "-q", "origin", BRANCH)
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, str(self.origin), str(self.copy)],
                       check=True)
        self.log: list[str] = []
        self.commands: list[list[str]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def sync(self, stale: bool = False) -> bool:
        with patch.object(sync, "run_logged",
                          lambda command, root, log: self.commands.append(command[1:])):
            return sync.sync_repository(BRANCH, self.log.append, self.copy, lambda: stale)

    def push(self, files: dict[str, str]) -> None:
        commit(self.author, files, "update")
        git(self.author, "push", "-q", "origin", BRANCH)

    def test_up_to_date_copy_does_nothing(self) -> None:
        self.assertFalse(self.sync())
        self.assertEqual(self.commands, [])
        self.assertTrue(any("already at" in line for line in self.log))

    def test_fast_forward_reinstalls_rebuilds_and_flags_gui_changes(self) -> None:
        self.push({"package-lock.json": '{"v": 2}', "benchmarks/bench-gui/server.py": "2"})
        self.assertTrue(self.sync(stale=True))
        self.assertEqual(git(self.copy, "rev-parse", "HEAD"), git(self.author, "rev-parse", "HEAD"))
        self.assertEqual(self.commands, [["ci"], ["run", "build:electron"]])

    def test_source_only_change_skips_npm_ci(self) -> None:
        self.push({"src/app.ts": "2"})
        self.assertFalse(self.sync(stale=True))
        self.assertEqual(self.commands, [["run", "build:electron"]])

    def test_local_edits_refuse_but_untracked_files_do_not(self) -> None:
        (self.copy / "notes.txt").write_text("untracked scratch")
        self.push({"src/app.ts": "2"})
        self.sync()
        (self.copy / "src" / "app.ts").write_text("edited on this machine")
        self.push({"src/app.ts": "3"})
        with self.assertRaisesRegex(sync.SyncError, "local changes \\(src/app.ts\\)"):
            self.sync()
        self.assertEqual((self.copy / "src" / "app.ts").read_text(), "edited on this machine")

    def test_diverged_history_is_never_rewritten(self) -> None:
        commit(self.copy, {"src/local.ts": "x"}, "local only")
        local_head = git(self.copy, "rev-parse", "HEAD")
        self.push({"src/app.ts": "2"})
        with self.assertRaisesRegex(sync.SyncError, "can't be fast-forwarded"):
            self.sync()
        self.assertEqual(git(self.copy, "rev-parse", "HEAD"), local_head)

    def test_another_checked_out_branch_refuses(self) -> None:
        git(self.copy, "checkout", "-q", "-b", "experiment")
        with self.assertRaisesRegex(sync.SyncError, "experiment checked out"):
            self.sync()


class RunnerSyncTests(unittest.TestCase):
    def runner(self) -> SimpleNamespace:
        import server
        log: list[str] = []
        fake = SimpleNamespace(log=log.append)
        return fake, lambda item: server.Runner.sync_before(fake, item), log

    def test_sync_is_skipped_unless_switched_on(self) -> None:
        import server
        _fake, sync_before, _log = self.runner()
        with patch.object(server, "load_settings", return_value=jobs.Settings()), \
                patch.object(server, "sync_repository") as sync_repository:
            self.assertTrue(sync_before("job"))
        sync_repository.assert_not_called()

    def test_refused_sync_fails_the_job_without_starting_it(self) -> None:
        import server
        _fake, sync_before, log = self.runner()
        spec = SimpleNamespace(state="queued", message="")
        settings = jobs.Settings(syncBeforeJobs="on", syncBranch="docling-beir")
        with patch.object(server, "load_settings", return_value=settings), \
                patch.object(server, "sync_repository",
                             side_effect=sync.SyncError("the copy has local changes (x)")), \
                patch.object(server, "load_job", return_value=spec), \
                patch.object(server, "save_job") as save_job:
            self.assertFalse(sync_before("job"))
        self.assertEqual(spec.state, "failed")
        self.assertIn("sync refused: the copy has local changes", spec.message)
        save_job.assert_called_once_with(spec)
        self.assertIn("job: Not started", log[-1])

    def test_branch_names_are_validated(self) -> None:
        for name in ("-rf", "main;rm", ""):
            with self.assertRaises(ValueError):
                jobs.Settings.from_dict({"syncBranch": name})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"syncBeforeJobs": "yes"})
        self.assertEqual(jobs.Settings.from_dict({"syncBranch": "feature/x-1"}).syncBranch, "feature/x-1")


if __name__ == "__main__":
    unittest.main()
