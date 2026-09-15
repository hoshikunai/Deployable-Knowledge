"""Process-only tests for the guarded HAKARI entry point."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "benchmarks" / "hakari" / "run_locked.py"


def process_is_live(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (OSError, IndexError):
        return False
    return state not in {"Z", "X"}


class HakariSupervisionTests(unittest.TestCase):
    def command(
        self,
        directory: Path,
        child_command: list[str],
        *extra: str,
    ) -> list[str]:
        return [
            sys.executable,
            str(WRAPPER),
            "--skip-memory-preflight",
            "--lease-path",
            str(directory / "lease"),
            "--meta-path",
            str(directory / "lease-owner.json"),
            "--status-path",
            str(directory / "status.json"),
            "--samples-path",
            str(directory / "samples.jsonl"),
            *extra,
            "--supervise-command",
            *child_command,
        ]

    def test_completed_command_writes_status_without_overwriting_lease_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = subprocess.run(
                self.command(directory, [sys.executable, "-c", "pass"]),
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((directory / "lease-owner.json").exists())
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"],
                "completed",
            )

    def test_memory_pause_terminates_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pid_path = directory / "child-pids.json"
            program = (
                "import json,subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"open({str(pid_path)!r},'w').write(json.dumps([child.pid])); "
                "time.sleep(30)"
            )
            result = subprocess.run(
                self.command(
                    directory,
                    [sys.executable, "-c", program],
                    "--memory-ceiling",
                    "1",
                    "--memory-warning",
                    "0",
                ),
                check=False,
                timeout=10,
            )

            self.assertEqual(result.returncode, 75)
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"],
                "memory-paused",
            )
            if pid_path.exists():
                for pid in json.loads(pid_path.read_text()):
                    self.assertFalse(process_is_live(pid))

    def test_timeout_is_distinct_from_memory_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = subprocess.run(
                self.command(
                    directory,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    "--timeout-seconds",
                    "0.1",
                ),
                check=False,
                timeout=10,
            )

            self.assertEqual(result.returncode, 124)
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"],
                "timed-out",
            )

    def test_sigterm_records_interruption_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            process = subprocess.Popen(
                self.command(
                    directory,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                )
            )
            samples_path = directory / "samples.jsonl"
            for _ in range(100):
                if samples_path.exists():
                    break
                time.sleep(0.02)
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=10)

            self.assertEqual(process.returncode, 130)
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"],
                "interrupted",
            )


if __name__ == "__main__":
    unittest.main()
