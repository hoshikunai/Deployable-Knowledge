"""Start, interrupt and stop benchmark process trees on Linux and Windows.

On Linux each child leads its own session, so signals reach its whole process group. Windows
has no process groups for signals: a child started in a new console process group receives
CTRL_BREAK_EVENT instead of SIGINT, and a stop kills the process tree found through psutil.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

import psutil

WINDOWS = sys.platform == "win32"


def new_group_kwargs() -> dict[str, Any]:
    """Popen arguments that give the child its own group, so stopping it spares the caller."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def interrupt_signals() -> tuple[int, ...]:
    """Signals a benchmark process turns into KeyboardInterrupt so its cleanup runs."""
    names = ("SIGTERM", "SIGINT", "SIGBREAK") if WINDOWS else ("SIGTERM", "SIGHUP", "SIGINT")
    return tuple(getattr(signal, name) for name in names)


def interrupt_group(pid: int) -> None:
    """Ask a child group to stop the way Ctrl+C would, letting it clean up first."""
    try:
        if WINDOWS:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(pid, signal.SIGINT)
    except (ProcessLookupError, OSError):
        return


def terminate_group(pgid: int, term_wait: float = 10, poll_interval: float = .1) -> None:
    """Stop a child group politely, then forcibly after term_wait seconds.

    On Windows the tree is captured before anything stops, because children whose parent
    has exited can no longer be found through it.
    """
    if WINDOWS:
        terminate_tree(pgid, term_wait)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + term_wait
    while time.monotonic() < deadline:
        if not group_alive(pgid):
            return
        time.sleep(poll_interval)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def group_alive(pgid: int) -> bool:
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == pgid and process.status() != psutil.STATUS_ZOMBIE:
                return True
        except (psutil.Error, ProcessLookupError, PermissionError):
            continue
    return False


def terminate_tree(pid: int, term_wait: float) -> None:
    try:
        root = psutil.Process(pid)
        tree = [*root.children(recursive=True), root]
    except psutil.NoSuchProcess:
        return
    interrupt_group(pid)
    _gone, alive = psutil.wait_procs(tree, timeout=term_wait)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=term_wait)
