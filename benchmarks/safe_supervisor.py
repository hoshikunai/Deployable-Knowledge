"""Generic process-group supervisor used by benchmark wrappers and tests."""
from __future__ import annotations
import argparse, os, signal, subprocess, time
from pathlib import Path
from benchmarks.benchmark_lease import BenchmarkLease, MemoryWatchdog, terminate_group, atomic_json

def main():
 p=argparse.ArgumentParser(); p.add_argument('--status',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--metrics',required=True); p.add_argument('--resume-command',required=True); p.add_argument('--ceiling',type=int,default=7*1024**3); p.add_argument('--timeout',type=float,default=300); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
 status=Path(a.status); checkpoint=Path(a.checkpoint); metrics=Path(a.metrics)
 child=None; reason='interrupted'
 def stop(sig, _frame):
  nonlocal reason
  reason='interrupted'; raise KeyboardInterrupt
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig, stop)
 with BenchmarkLease(runName='supervised-fake'):
  child=subprocess.Popen(a.command, start_new_session=True)
  try:
   watch=MemoryWatchdog(child.pid, ceiling=a.ceiling, warning=max(1,a.ceiling-1), resume_command=a.resume_command, status_path=status, samples_path=status.with_suffix('.samples.jsonl'))
   deadline=time.monotonic()+a.timeout
   while child.poll() is None:
    if watch.poll()=='memory-paused': reason='memory-paused'; raise RuntimeError(reason)
    if time.monotonic()>=deadline: reason='timed-out'; raise TimeoutError(reason)
    time.sleep(.05)
   if child.returncode: raise RuntimeError('failed')
   atomic_json(status, {'status':'completed','reason':'completed','peakOwnedRss':watch.peak,'lastSample':watch.last,'childIdentities':[child.pid],'checkpointCount':1})
  except TimeoutError:
   terminate_group(child.pid)
   try: child.wait(timeout=2)
   except subprocess.TimeoutExpired: pass
   atomic_json(status, {'status':'timed-out','reason':reason,'peakOwnedRss':watch.peak,'lastSample':watch.last,'childIdentities':[child.pid],'checkpointCount':1,'resumeCommand':a.resume_command})
  except KeyboardInterrupt:
   terminate_group(child.pid)
   try: child.wait(timeout=2)
   except subprocess.TimeoutExpired: pass
   atomic_json(status, {'status':'interrupted','reason':reason,'peakOwnedRss':watch.peak,'lastSample':watch.last,'childIdentities':[child.pid],'checkpointCount':1,'resumeCommand':a.resume_command})
  except RuntimeError as exc:
   terminate_group(child.pid)
   try: child.wait(timeout=2)
   except subprocess.TimeoutExpired: pass
   atomic_json(status, {'status':str(exc),'reason':str(exc),'peakOwnedRss':watch.peak,'lastSample':watch.last,'childIdentities':[child.pid],'checkpointCount':1,'resumeCommand':a.resume_command})

if __name__=='__main__': main()
