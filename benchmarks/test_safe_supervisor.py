import json, os, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path

class SupervisorTests(unittest.TestCase):
 def run_case(self, sig=None, timeout=5, ceiling=10**12):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); st=d/'status.json'; cp=d/'checkpoint'; mt=d/'metrics'; cp.write_text('checkpoint')
   cmd=[sys.executable,'-m','benchmarks.safe_supervisor','--status',str(st),'--checkpoint',str(cp),'--metrics',str(mt),'--resume-command','resume','--ceiling',str(ceiling),'--timeout',str(timeout),sys.executable,'-c','import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); time.sleep(30)']
   proc=subprocess.Popen(cmd, start_new_session=True); time.sleep(.3)
   if sig: os.kill(proc.pid,sig)
   try:
    proc.wait(timeout=25)
   finally:
    if proc.poll() is None: proc.kill(); proc.wait()
   data=json.loads(st.read_text()); self.assertEqual(data['status'], 'memory-paused' if ceiling<10**6 else ('timed-out' if timeout<1 else 'interrupted')); self.assertTrue(cp.exists()); self.assertFalse(mt.exists()); self.assertIn('resumeCommand',data)
 def test_sigterm(self): self.run_case(signal.SIGTERM)
 def test_sighup(self): self.run_case(signal.SIGHUP)
 def test_timeout(self): self.run_case(timeout=.1)
 def test_memory(self): self.run_case(ceiling=1)
if __name__=='__main__': unittest.main()
