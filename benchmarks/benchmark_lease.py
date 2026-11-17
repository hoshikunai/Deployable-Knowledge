"""Exclusive, memory-aware supervision primitives for local benchmark runs."""
from __future__ import annotations

import fcntl, json, os, signal, time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASE_PATH = ROOT / ".cache" / "benchmark-lease.lock"
META_PATH = ROOT / ".cache" / "benchmark-lease.json"
SAMPLES_PATH = ROOT / ".cache" / "benchmark-memory-samples.jsonl"
SCHEMA = 1

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
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (OSError, IndexError):
        return None

def rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0

def descendants(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            data = (proc / "stat").read_text().split()
            children.setdefault(int(data[3]), []).append(int(data[0]))
        except (OSError, ValueError, IndexError):
            continue
    result: list[int] = []
    queue = [root_pid]
    while queue:
        parent = queue.pop(0)
        for child in children.get(parent, []):
            result.append(child); queue.append(child)
    return result

def tree_rss(root_pid) -> tuple[int, list[int]]:
    roots = [root_pid] if isinstance(root_pid, int) else list(root_pid)
    pids = list(dict.fromkeys(pid for root in roots for pid in [root] + descendants(root)))
    return sum(rss_bytes(pid) for pid in pids), pids

def memory_preflight(min_available=4*1024**3, min_swap=2*1024**3) -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if value.strip().endswith(" kB"):
            values[key] = int(value.split()[0]) * 1024
    result = {"memAvailable": values.get("MemAvailable", 0), "swapFree": values.get("SwapFree", 0),
              "minMemAvailable": min_available, "minSwapFree": min_swap}
    if result["memAvailable"] < min_available or result["swapFree"] < min_swap:
        raise RuntimeError(f"insufficient memory headroom: {result}")
    return result

def terminate_group(pgid: int, term_wait=10, poll_interval=.1) -> None:
    try: os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError: return
    deadline = time.monotonic() + term_wait
    while time.monotonic() < deadline:
        live = False
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                fields=(proc/'stat').read_text().split()
                if int(fields[4]) == pgid and fields[2] not in ('Z','X'):
                    live=True; break
            except (OSError, ValueError, IndexError): pass
        if not live: return
        time.sleep(poll_interval)
    try: os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError: pass

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
        self.read_memory = read_memory or self._read_memory
        self.global_warning_mem, self.global_stop_mem = global_warning_mem, global_stop_mem
        self.global_warning_swap, self.global_stop_swap = global_warning_swap, global_stop_swap
        self.last = None
    @staticmethod
    def _read_memory():
        values={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,_,val=line.partition(':')
            if val.strip().endswith(' kB'): values[key]=int(val.split()[0])*1024
        return values.get('MemAvailable',0), values.get('SwapFree',0)
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
    if not isinstance(pgid, int) or os.getpgid(metadata["pid"]) != pgid: return False
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

class BenchmarkLease:
    def __init__(self, *, lease_path=None, meta_path=None, **metadata):
        self.lease_path = Path(lease_path) if lease_path is not None else LEASE_PATH
        self.meta_path = Path(meta_path) if meta_path is not None else META_PATH
        self.metadata = metadata; self.fd = None
    def __enter__(self):
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lease_path, "a+")
        try: fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fd.close(); self.fd = None
            raise RuntimeError(f"blocked-active-run: {self.meta_path}")
        self.metadata = {"schemaVersion": SCHEMA, "pid": os.getpid(),
                         "startIdentity": start_identity(os.getpid()),
                         "pgid": os.getpgid(os.getpid()),
                         "command": " ".join(os.sys.argv), "createdAt": time.time(),
                         "heartbeatAt": time.time(), **self.metadata}
        atomic_json(self.meta_path, self.metadata)
        return self
    def heartbeat(self, **extra):
        self.metadata.update(extra, heartbeatAt=time.time()); atomic_json(self.meta_path, self.metadata)
    def __exit__(self, *_):
        try:
            if self.meta_path.exists() and json.loads(self.meta_path.read_text()).get("pid") == os.getpid(): self.meta_path.unlink()
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close(); self.fd = None

@contextmanager
def lease(**metadata):
    with BenchmarkLease(**metadata) as owner: yield owner
