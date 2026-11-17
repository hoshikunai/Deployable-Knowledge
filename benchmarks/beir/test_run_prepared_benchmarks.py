from __future__ import annotations

import unittest
import runpy
from pathlib import Path
from unittest.mock import patch

import run_prepared_benchmarks as runner


class PreparedBenchmarkRunnerTests(unittest.TestCase):
    def test_supervisor_accepts_all_prepared_datasets(self) -> None:
        supervisor_datasets = runpy.run_path(str(runner.PLAN))["DATASETS"]
        self.assertLessEqual(set(runner.PREPARED_MAPPINGS), set(supervisor_datasets))

    def test_resume_command_keeps_mapping_and_timeout(self) -> None:
        command = runner.plan_command(
            "arguana", "mapping.json", "run-1", 45, resume=True
        )
        self.assertEqual(command[-1], "--resume")
        self.assertEqual(command[command.index("--reuse-document-mapping") + 1], "mapping.json")
        self.assertEqual(command[command.index("--timeout-minutes") + 1], "45")
        self.assertEqual(command[command.index("--search-depth") + 1], "10")

    def test_document_depth_is_forwarded_to_supervisor(self) -> None:
        command = runner.plan_command(
            "scifact", "mapping.json", "run-100", 45, resume=False,
            document_depth=100,
        )
        self.assertEqual(command[command.index("--search-depth") + 1], "100")

    def test_mapping_overrides_reject_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "twice"):
            runner.mapping_overrides(["arguana=first.json", "arguana=second.json"])

    def test_non_finite_metric_is_rejected(self) -> None:
        self.assertFalse(runner.finite_metrics({"hybrid": {"NDCG@10": float("nan")}}))
        self.assertFalse(runner.finite_metrics({"hybrid": {"NDCG@10": "0.4"}}))

    def test_timeout_with_progress_resumes_once(self) -> None:
        statuses = [
            {"status": "timed-out", "checkpointCount": 10},
            {"status": "completed", "checkpointCount": 20},
        ]
        with (
            patch("benchmarks.benchmark_lease.memory_preflight"),
            patch.object(runner, "run_command", side_effect=[1, 0]) as run_command,
            patch.object(runner, "read_status", side_effect=statuses),
            patch.object(runner, "verify_completed_run", return_value=Path("metrics.json")),
        ):
            result = runner.run_dataset("arguana", "mapping.json", "run-1", 45, 2, False)
        self.assertEqual(result, Path("metrics.json"))
        self.assertNotIn("--resume", run_command.call_args_list[0].args[0])
        self.assertIn("--resume", run_command.call_args_list[1].args[0])

    def test_timeout_without_progress_stops(self) -> None:
        with (
            patch("benchmarks.benchmark_lease.memory_preflight"),
            patch.object(runner, "run_command", return_value=1),
            patch.object(runner, "read_status", return_value={"status": "timed-out", "checkpointCount": 0}),
        ):
            with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                runner.run_dataset("arguana", "mapping.json", "run-1", 45, 8, False)

    def test_early_supervisor_error_is_not_reported_as_missing_status(self) -> None:
        with (
            patch("benchmarks.benchmark_lease.memory_preflight"),
            patch.object(runner, "run_command", return_value=2),
            patch.object(runner, "read_status", side_effect=RuntimeError("missing status")),
        ):
            with self.assertRaisesRegex(RuntimeError, "supervisor exited 2"):
                runner.run_dataset("scifact", "mapping.json", "run-1", 45, 1, False)


if __name__ == "__main__":
    unittest.main()
