"""Cross-process lease tests for the BEIR and HAKARI entry points."""
import json, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

class CrossRunnerLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.lease, self.meta, self.sentinel = base/'lease', base/'meta.json', base/'sentinel'

    def tearDown(self): self.tmp.cleanup()

    def invoke(self, wrapper, command, hold=False):
        prefix = ['--execute'] if 'full_beir_plan.py' in wrapper else []
        args = [PYTHON, str(ROOT / wrapper), *prefix, '--skip-memory-preflight', '--lease-path', str(self.lease), '--meta-path', str(self.meta), '--supervise-command', *command]
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) if hold else subprocess.run(args, capture_output=True, text=True)

    def test_hakari_blocks_beir_and_then_both_release(self):
        holder = self.invoke('benchmarks/hakari/run_locked.py', [PYTHON, '-c', 'import time; time.sleep(1.5)'], True)
        for _ in range(30):
            if self.meta.exists(): break
            time.sleep(.05)
        blocked = self.invoke('benchmarks/rag-evaluation/full_beir_plan.py', [PYTHON, '-c', f'open({str(self.sentinel)!r}, "w")'])
        self.assertNotEqual(blocked.returncode, 0); self.assertIn('blocked-active-run', blocked.stderr); self.assertFalse(self.sentinel.exists())
        holder.wait(timeout=5); self.assertFalse(self.meta.exists())
        done = self.invoke('benchmarks/rag-evaluation/full_beir_plan.py', [PYTHON, '-c', f'open({str(self.sentinel)!r}, "w")'])
        self.assertEqual(done.returncode, 0); self.assertTrue(self.sentinel.exists())

    def test_beir_blocks_hakari_and_metadata_owner(self):
        holder = self.invoke('benchmarks/rag-evaluation/full_beir_plan.py', [PYTHON, '-c', 'import time; time.sleep(1.5)'], True)
        for _ in range(30):
            if self.meta.exists(): break
            time.sleep(.05)
        metadata = json.loads(self.meta.read_text()); self.assertIn('command', metadata); self.assertEqual(metadata['pid'], holder.pid)
        blocked = self.invoke('benchmarks/hakari/run_locked.py', [PYTHON, '-c', f'open({str(self.sentinel)!r}, "w")'])
        self.assertNotEqual(blocked.returncode, 0); self.assertIn('blocked-active-run', blocked.stderr); self.assertFalse(self.sentinel.exists())
        holder.wait(timeout=5); self.assertFalse(self.meta.exists())
        done = self.invoke('benchmarks/hakari/run_locked.py', [PYTHON, '-c', f'open({str(self.sentinel)!r}, "w")'])
        self.assertEqual(done.returncode, 0); self.assertTrue(self.sentinel.exists())

if __name__ == '__main__': unittest.main()
