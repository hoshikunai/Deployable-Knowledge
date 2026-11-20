"""Exclusive, memory-aware supervision primitives for local benchmark runs."""
from __future__ import annotations

import json, os, sys, time
from contextlib import contextmanager
from pathlib import Path

import psutil

from benchmarks.process_group import WINDOWS, terminate_group

if WINDOWS:
    import msvcrt
else:
    import fcntl

ROOT = Path(__file__).resolve().parents[1]
LEASE_PATH = ROOT / ".cache" / "benchmark-lease.lock"
META_PATH = ROOT / ".cache" / "benchmark-lease.json"
SAMPLES_PATH = ROOT / ".cache" / "benchmark-memory-samples.jsonl"
SCHEMA = 1
# Prepares parse whole PDF corpora; query runs peak near 1.5 GB and the watchdog still stops
# any job below 2 GB available.
PREPARE_MIN_AVAILABLE = 4 * 1024**3
QUERY_MIN_AVAILABLE = int(2.5 * 1024**3)

class SupervisionError(RuntimeError):
    status = "failed"
    def __init__(self, reason, *, peak_owned_rss=None, last_sample=None,
                 child_identities=None, checkpoint_count=0, resume_command=None):
        super().__init__(reason)
        self.reason = reason
        self.peakOwnedRss = peak_owned_rss
        self.lastSample = last_sample
        self.childIdentities = child_identities or []
        self.checkpointCount = checkpoint_count
        self.resumeCommand = resume_command

class TimedOutError(SupervisionError):
    status = "timed-out"

class MemoryPausedError(SupervisionError):
    status = "memory-paused"

class InterruptedError(SupervisionError):
    status = "interrupted"

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def start_identity(pid: int) -> str | None:
    """A process's start time, which tells a live owner apart from a reused PID."""
    try:
        return repr(psutil.Process(pid).create_time())
    except psutil.Error:
        return None

def rss_bytes(pid: int) -> int:
    try:
        return psutil.Process(pid).memory_info().rss
    except psutil.Error:
        return 0

def descendants(root_pid: int) -> list[int]:
    try:
        return [child.pid for child in psutil.Process(root_pid).children(recursive=True)]
    except psutil.Error:
        return []

def tree_rss(root_pid) -> tuple[int, list[int]]:
    roots = [root_pid] if isinstance(root_pid, int) else list(root_pid)
    pids = list(dict.fromkeys(pid for root in roots for pid in [root] + descendants(root)))
    return sum(rss_bytes(pid) for pid in pids), pids

def system_memory() -> tuple[int, int]:
    """Available memory and free swap (MemAvailable/SwapFree; the page file on Windows)."""
    return psutil.virtual_memory().available, psutil.swap_memory().free

def memory_preflight(min_available=PREPARE_MIN_AVAILABLE, min_swap=2*1024**3) -> dict:
    available, swap = system_memory()
    result = {"memAvailable": available, "swapFree": swap,
              "minMemAvailable": min_available, "minSwapFree": min_swap}
    if result["memAvailable"] < min_available or result["swapFree"] < min_swap:
        raise RuntimeError(f"insufficient memory headroom: {result}")
    return result

class MemoryWatchdog:
    """Poll an owned process tree; callers stop work when ``poll`` returns true."""
    def __init__(self, root_pid, ceiling=7*1024**3, warning=6*1024**3,
                 interval=5, status_path=None, samples_path=None,
                 resume_command=None, read_memory=None,
                 global_warning_mem=3*1024**3, global_stop_mem=2*1024**3,
                 global_warning_swap=2*1024**3, global_stop_swap=1*1024**3):
        if status_path is None or samples_path is None: raise ValueError('explicit status_path and samples_path required')
        self.root_pid, self.ceiling, self.warning = root_pid, ceiling, warning
        self.interval, self.status_path, self.samples_path = interval, Path(status_path), Path(samples_path)
        self.resume_command, self.peak = resume_command, 0
        self.read_memory = read_memory or system_memory
        self.global_warning_mem, self.global_stop_mem = global_warning_mem, global_stop_mem
        self.global_warning_swap, self.global_stop_swap = global_warning_swap, global_stop_swap
        self.last = None
    def poll(self):
        total, pids = tree_rss(self.root_pid); self.peak = max(self.peak, total)
        available, swap = self.read_memory()
        sample = {"timestamp": time.time(), "rssBytes": total, "pids": pids, "peakRssBytes": self.peak, "memAvailable": available, "swapFree": swap}
        self.last = sample; self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        with self.samples_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(sample) + "\n")
        reason = None
        if total >= self.ceiling: reason = 'owned RSS exceeded ceiling'
        elif available < self.global_stop_mem: reason = 'MemAvailable below stop threshold'
        elif swap < self.global_stop_swap: reason = 'SwapFree below stop threshold'
        if reason:
            atomic_json(self.status_path, {"status":"memory-paused", "reason":reason, "peakOwnedRss":self.peak, "lastSample":sample,
                                           "childIdentities":[], "checkpointCount":0,
                                           "resumeCommand":self.resume_command})
            return "memory-paused"
        if total >= self.warning or available < self.global_warning_mem or swap < self.global_warning_swap:
            return "warning"
        return "ok"

def verified_owner(metadata: dict) -> bool:
    pid = metadata.get("pid"); identity = metadata.get("startIdentity")
    return isinstance(pid, int) and start_identity(pid) == identity and metadata.get("command")

def cleanup_verified(metadata: dict) -> bool:
    if not verified_owner(metadata): return False
    pgid = metadata.get("pgid")
    if not isinstance(pgid, int): return False
    if not WINDOWS and os.getpgid(metadata["pid"]) != pgid: return False
    terminate_group(pgid); return True

def scan_markers(roots) -> list[dict]:
    """Report marker ownership; never signals during scanning."""
    found=[]
    for root in map(Path, roots):
        if not root.exists(): continue
        for path in root.rglob('benchmark-owner.json'):
            try:
                item=json.loads(path.read_text()); item['path']=str(path)
                runtime=Path(item.get('runtimePath','')).resolve()
                item['verified']=bool(item.get('pgid') and item.get('command') and verified_owner(item) and runtime == path.parent.resolve())
                found.append(item)
            except (OSError, json.JSONDecodeError): found.append({'path':str(path),'verified':False,'reason':'malformed'})
    return found

VALID_STATUSES={'completed','failed','timed-out','memory-paused','interrupted','blocked-active-run'}
def validate_status(value: dict) -> None:
    if value.get('status') not in VALID_STATUSES: raise ValueError('invalid status')
    for key in ('reason','peakOwnedRss','lastSample','childIdentities','checkpointCount'):
        if key not in value: raise ValueError(f'missing {key}')
    if value['status'] in {'timed-out','memory-paused','interrupted'} and 'resumeCommand' not in value:
        raise ValueError('missing resumeCommand')
    if value['status']=='blocked-active-run' and 'activeOwner' not in value:
        raise ValueError('missing activeOwner')

def lock_exclusive(handle) -> bool:
    """Take a non-blocking exclusive lock on an open file; False if another process holds it."""
    try:
        if WINDOWS:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError, OSError):
        return False
    return True

def unlock(handle) -> None:
    if WINDOWS:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)

class BenchmarkLease:
    def __init__(self, *, lease_path=None, meta_path=None, **metadata):
        self.lease_path = Path(lease_path) if lease_path is not None else LEASE_PATH
        self.meta_path = Path(meta_path) if meta_path is not None else META_PATH
        self.metadata = metadata; self.fd = None
    def __enter__(self):
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lease_path, "a+")
        if not lock_exclusive(self.fd):
            self.fd.close(); self.fd = None
            raise RuntimeError(f"blocked-active-run: {self.meta_path}")
        self.metadata = {"schemaVersion": SCHEMA, "pid": os.getpid(),
                         "startIdentity": start_identity(os.getpid()),
                         "pgid": os.getpid() if WINDOWS else os.getpgid(os.getpid()),
                         "command": " ".join(sys.argv), "createdAt": time.time(),
                         "heartbeatAt": time.time(), **self.metadata}
        atomic_json(self.meta_path, self.metadata)
        return self
    def heartbeat(self, **extra):
        self.metadata.update(extra, heartbeatAt=time.time()); atomic_json(self.meta_path, self.metadata)
    def __exit__(self, *_):
        try:
            if self.meta_path.exists() and json.loads(self.meta_path.read_text()).get("pid") == os.getpid(): self.meta_path.unlink()
        finally:
            unlock(self.fd); self.fd.close(); self.fd = None

@contextmanager
def lease(**metadata):
    with BenchmarkLease(**metadata) as owner: yield owner
