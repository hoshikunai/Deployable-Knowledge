"""Small supervision contract tests; no servers, sockets, or models are started."""
import unittest

from benchmarks.benchmark_lease import (
    InterruptedError,
    MemoryPausedError,
    TimedOutError,
    validate_status,
)


class TypedSupervisionTests(unittest.TestCase):
    def assert_status(self, exception, status):
        error = exception("reason", peak_owned_rss=12, last_sample={"rssBytes": 12},
                          child_identities=["child"], checkpoint_count=3,
                          resume_command="resume")
        self.assertEqual(error.status, status)
        self.assertEqual(error.peakOwnedRss, 12)
        self.assertEqual(error.lastSample, {"rssBytes": 12})
        self.assertEqual(error.childIdentities, ["child"])
        self.assertEqual(error.checkpointCount, 3)
        self.assertEqual(error.resumeCommand, "resume")

    def test_memory_exception(self):
        self.assert_status(MemoryPausedError, "memory-paused")

    def test_timeout_exception(self):
        self.assert_status(TimedOutError, "timed-out")

    def test_signal_exception(self):
        self.assert_status(InterruptedError, "interrupted")

    def test_incomplete_status_contract(self):
        for status in ("timed-out", "memory-paused", "interrupted"):
            validate_status({"status": status, "reason": "x", "peakOwnedRss": None,
                             "lastSample": None, "childIdentities": [],
                             "checkpointCount": 2, "resumeCommand": "resume"})


if __name__ == "__main__":
    unittest.main()
