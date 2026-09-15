"""Gated, resumable full-BEIR runner; dry-run is side-effect free."""
from __future__ import annotations
import argparse, hashlib, json, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_lease import (BenchmarkLease, memory_preflight,
    SupervisionError, InterruptedError, validate_status)
DATASETS = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'trec-covid', 'hotpotqa']

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def checkpoint_count(run_directory):
    path = run_directory / 'query-checkpoints.jsonl'
    if not path.is_file(): return 0
    try:
        return sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    except OSError:
        return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--execute', action='store_true')
    p.add_argument('--status', action='store_true'); p.add_argument('--cleanup-stale', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--run-name'); p.add_argument('--split', default='test'); p.add_argument('--search-depth', type=int, default=10); p.add_argument('--timeout-minutes', type=int, default=45)
    p.add_argument('--supervise-command', nargs=argparse.REMAINDER); p.add_argument('--lease-path'); p.add_argument('--meta-path'); p.add_argument('--skip-memory-preflight', action='store_true')
    a = p.parse_args()
    if a.status:
        from benchmarks import benchmark_lease
        print(json.dumps({'lease': str(benchmark_lease.LEASE_PATH), 'metadata': json.loads(benchmark_lease.META_PATH.read_text()) if benchmark_lease.META_PATH.exists() else None}, indent=2)); return
    if a.cleanup_stale:
        print('Refusing implicit stale cleanup: inspect --status and terminate only verified owned groups.'); return
    if not a.execute:
        print(json.dumps({'status':'dry-run','startsServer':False,'issuesSearch':False,'commands':[f'python benchmarks/rag-evaluation/full_beir_plan.py --execute --dataset {d} --split test' for d in DATASETS]}, indent=2)); return
    if a.supervise_command:
        if not a.skip_memory_preflight: memory_preflight()
        with BenchmarkLease(lease_path=a.lease_path, meta_path=a.meta_path, runName='beir-supervised', dataset='beir', runtimePath='benchmarks/beir'):
            raise SystemExit(subprocess.run(a.supervise_command).returncode)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'beir'))
    import run_suite
    if not a.dataset: raise SystemExit('--dataset is required with --execute')
    if a.resume and not a.run_name: raise SystemExit('--run-name is required with --resume')
    run_name = a.run_name or f"{a.dataset}-full-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
    out = ROOT / 'benchmarks' / 'rag-evaluation' / 'runs' / run_name
    config = {'dataset':a.dataset,'split':a.split,'searchDepth':a.search_depth,'timeoutMinutes':a.timeout_minutes,'runName':run_name}
    cfg_path = out / 'config.json'
    if a.resume:
        if not cfg_path.is_file(): raise SystemExit(f'cannot resume missing config: {cfg_path}')
        saved = json.loads(cfg_path.read_text())
        if fingerprint(saved) != fingerprint(config): raise SystemExit('resume configuration mismatch')
    elif out.exists(): raise SystemExit(f'refusing to overwrite existing run: {out}')
    else:
        write_atomic(cfg_path, config); write_atomic(out/'status.json', {'status':'started','configFingerprint':fingerprint(config)})
    suite_args = argparse.Namespace(split=a.split, search_depth=a.search_depth, port=4179, startup_timeout=120, no_shared_model_cache=False, query_counts='all', sample_seed=42, run_name=run_name, resume=a.resume, runtime_id=f'runtime-{a.dataset}-001', timeout_seconds=a.timeout_minutes * 60)
    if not a.skip_memory_preflight: memory_preflight()
    started = time.monotonic()
    def deadline(_signum, _frame): raise TimeoutError(f'corpus exceeded {a.timeout_minutes} minute cap')
    def interrupted(signum, _frame): raise InterruptedError(f'interrupted by signal {signum}', resume_command=resume_command)
    resume_command = f'python benchmarks/rag-evaluation/full_beir_plan.py --execute --resume --dataset {a.dataset} --run-name {run_name}'
    previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    signal.signal(signal.SIGALRM, deadline)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT): signal.signal(s, interrupted)
    signal.alarm(a.timeout_minutes * 60)
    try:
      with BenchmarkLease(lease_path=a.lease_path, meta_path=a.meta_path, runName=run_name, dataset=a.dataset, runtimePath=str(ROOT / '.cache' / 'beir' / suite_args.runtime_id), port=suite_args.port, memoryCeilingBytes=7*1024**3) as owner:
        result = run_suite.run_dataset(a.dataset, suite_args, run_name, owner)
        completed = {'status':'completed','configFingerprint':fingerprint(config),
          'elapsedSeconds':time.monotonic()-started,'reason':None,'peakOwnedRss':None,
          'lastSample':None,'childIdentities':[],'checkpointCount':checkpoint_count(out)}
        validate_status(completed)
        write_atomic(out/'result.json', result); write_atomic(out/'status.json', completed)
    except Exception as exc:
        status = exc.status if isinstance(exc, SupervisionError) else ('timed-out' if isinstance(exc, TimeoutError) else 'failed')
        record = {'status':status,'configFingerprint':fingerprint(config),'reason':str(exc),
          'peakOwnedRss':getattr(exc,'peakOwnedRss',None),'lastSample':getattr(exc,'lastSample',None),
          'childIdentities':getattr(exc,'childIdentities',[]),'checkpointCount':checkpoint_count(out)}
        if status != 'completed': record['resumeCommand'] = resume_command
        validate_status(record)
        write_atomic(out/'status.json', record)
        raise
    finally:
        signal.alarm(0)
        for s, handler in previous.items(): signal.signal(s, handler)

if __name__ == '__main__': main()
