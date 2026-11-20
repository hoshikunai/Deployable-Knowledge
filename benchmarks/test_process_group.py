import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import psutil

from benchmarks import process_group

PARENT_WITH_STUBBORN_CHILD = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"],
    stdout=subprocess.PIPE, text=True)
child.stdout.readline()
print(child.pid, flush=True)
time.sleep(60)
"""


class ProcessGroupTests(unittest.TestCase):
    def test_group_arguments_per_platform(self):
        with patch.object(process_group, "WINDOWS", False):
            self.assertEqual(process_group.new_group_kwargs(), {"start_new_session": True})
        with patch.object(process_group, "WINDOWS", True), \
                patch.object(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True):
            self.assertEqual(process_group.new_group_kwargs(), {"creationflags": 0x200})

    def test_interrupt_signals_per_platform(self):
        with patch.object(process_group, "WINDOWS", False), \
                patch.object(signal, "SIGHUP", 1, create=True):
            self.assertEqual(process_group.interrupt_signals(), (signal.SIGTERM, 1, signal.SIGINT))
        with patch.object(process_group, "WINDOWS", True), \
                patch.object(signal, "SIGBREAK", 21, create=True):
            self.assertEqual(process_group.interrupt_signals(), (signal.SIGTERM, signal.SIGINT, 21))

    def test_windows_interrupt_sends_ctrl_break(self):
        with patch.object(process_group, "WINDOWS", True), \
                patch.object(signal, "CTRL_BREAK_EVENT", 1, create=True), \
                patch.object(process_group.os, "kill") as kill:
            process_group.interrupt_group(1234)
        kill.assert_called_once_with(1234, 1)

    def test_terminate_group_stops_a_session(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 **process_group.new_group_kwargs())
        process_group.terminate_group(child.pid, term_wait=5)
        self.assertIsNotNone(child.wait(timeout=5))
        process_group.terminate_group(child.pid)

    def test_tree_stop_kills_children_that_ignore_the_interrupt(self):
        parent = subprocess.Popen([sys.executable, "-c", PARENT_WITH_STUBBORN_CHILD],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  **process_group.new_group_kwargs())
        child_pid = int(parent.stdout.readline())
        started = time.monotonic()
        process_group.terminate_tree(parent.pid, term_wait=1)
        parent.wait(timeout=5)
        self.assertFalse(psutil.pid_exists(child_pid) and
                         psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE)
        self.assertLess(time.monotonic() - started, 5)
        process_group.terminate_tree(parent.pid, term_wait=1)


if __name__ == "__main__":
    unittest.main()
