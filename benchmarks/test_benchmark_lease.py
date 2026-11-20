import json, os, signal, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks import benchmark_lease as lease

class LeaseTests(unittest.TestCase):
    def _assert_memory_paused_status(self, path):
        value = json.loads(path.read_text())
        self.assertEqual(value['status'], 'memory-paused')
        lease.validate_status(value)
        self.assertIn('MemAvailable', value['reason']) if 'MemAvailable' in value['reason'] else self.assertIn('SwapFree', value['reason'])

    def test_exclusive_and_release(self):
        with tempfile.TemporaryDirectory() as d, patch.object(lease, 'LEASE_PATH', Path(d)/'l'), patch.object(lease, 'META_PATH', Path(d)/'m'):
            with lease.BenchmarkLease(runName='a'):
                with self.assertRaises(RuntimeError): lease.BenchmarkLease(runName='b').__enter__()
            with lease.BenchmarkLease(runName='c'): pass
    def test_rss_descendant(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; x=bytearray(1024*1024); time.sleep(2)'])
        try: total, pids = lease.tree_rss(os.getpid()); self.assertIn(os.getpid(), pids); self.assertGreaterEqual(total, 0)
        finally: child.terminate(); child.wait()
    def test_stale_reused_pid_not_killed(self):
        self.assertFalse(lease.verified_owner({'pid': os.getpid(), 'startIdentity': 'wrong', 'command':'benchmark'}))
    def test_start_identity_is_stable_and_absent_for_dead_pids(self):
        self.assertEqual(lease.start_identity(os.getpid()), lease.start_identity(os.getpid()))
        child = subprocess.Popen([sys.executable, '-c', 'pass']); child.wait()
        self.assertIsNone(lease.start_identity(child.pid))
        self.assertEqual(lease.rss_bytes(child.pid), 0)
        self.assertEqual(lease.descendants(child.pid), [])

    def test_memory_preflight_uses_system_memory(self):
        with patch.object(lease, 'system_memory', return_value=(8*1024**3, 4*1024**3)):
            self.assertEqual(lease.memory_preflight()['memAvailable'], 8*1024**3)
        with patch.object(lease, 'system_memory', return_value=(1024**3, 4*1024**3)):
            with self.assertRaisesRegex(RuntimeError, 'insufficient memory'):
                lease.memory_preflight()
        available, swap = lease.system_memory()
        self.assertGreater(available, 0); self.assertGreaterEqual(swap, 0)

    def test_lock_is_exclusive_until_released(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'lock'
            with open(path, 'a+') as first, open(path, 'a+') as second:
                self.assertTrue(lease.lock_exclusive(first))
                probe = subprocess.run([sys.executable, '-c',
                    'import sys; sys.path.insert(0, sys.argv[2]); from benchmarks import benchmark_lease as l; '
                    'print(l.lock_exclusive(open(sys.argv[1], "a+")))', str(path), str(Path(lease.ROOT))],
                    capture_output=True, text=True, check=True)
                self.assertEqual(probe.stdout.strip(), 'False')
                lease.unlock(first)
                self.assertTrue(lease.lock_exclusive(second))
                lease.unlock(second)

    def test_watchdog_transition(self):
        with tempfile.TemporaryDirectory() as d:
            w=lease.MemoryWatchdog(os.getpid(), ceiling=1, warning=1, samples_path=Path(d)/'s', status_path=Path(d)/'m', resume_command='resume')
            self.assertEqual(w.poll(), 'memory-paused'); self.assertEqual(json.loads((Path(d)/'m').read_text())['status'], 'memory-paused')

    def test_global_memory_warning_with_owned_rss_healthy(self):
        with tempfile.TemporaryDirectory() as d:
            w = lease.MemoryWatchdog(os.getpid(), ceiling=10**15, warning=10**15,
                samples_path=Path(d)/'samples', status_path=Path(d)/'status',
                read_memory=lambda: (3*1024**3 - 1, 2*1024**3),
                global_warning_mem=3*1024**3, global_stop_mem=2*1024**3)
            self.assertEqual(w.poll(), 'warning')
            sample = json.loads((Path(d)/'samples').read_text())
            self.assertEqual(sample['memAvailable'], 3*1024**3 - 1)
            self.assertEqual(sample['swapFree'], 2*1024**3)

    def test_global_memory_stop(self):
        with tempfile.TemporaryDirectory() as d:
            status = Path(d)/'status'
            w = lease.MemoryWatchdog(os.getpid(), ceiling=10**15, warning=10**15,
                samples_path=Path(d)/'samples', status_path=status,
                read_memory=lambda: (2*1024**3 - 1, 2*1024**3))
            self.assertEqual(w.poll(), 'memory-paused')
            self._assert_memory_paused_status(status)

    def test_global_swap_stop(self):
        with tempfile.TemporaryDirectory() as d:
            status = Path(d)/'status'
            w = lease.MemoryWatchdog(os.getpid(), ceiling=10**15, warning=10**15,
                samples_path=Path(d)/'samples', status_path=status,
                read_memory=lambda: (8*1024**3, 1024**3 - 1))
            self.assertEqual(w.poll(), 'memory-paused')
            self._assert_memory_paused_status(status)

    def test_healthy_global_and_owned_memory(self):
        with tempfile.TemporaryDirectory() as d:
            w = lease.MemoryWatchdog(os.getpid(), ceiling=10**15, warning=10**15,
                samples_path=Path(d)/'samples', status_path=Path(d)/'status',
                read_memory=lambda: (8*1024**3, 4*1024**3))
            self.assertEqual(w.poll(), 'ok')

    def test_multiple_roots_combined_limit(self):
        groups = []
        try:
            for _ in range(2):
                groups.append(subprocess.Popen([sys.executable, '-c',
                    'import time; data=bytearray(8*1024*1024); time.sleep(10)'], start_new_session=True))
            import time; time.sleep(.2)
            individual = [lease.tree_rss(p.pid)[0] for p in groups]
            ceiling = max(individual) + 1
            with tempfile.TemporaryDirectory() as d:
                w = lease.MemoryWatchdog([p.pid for p in groups], ceiling=ceiling, warning=10**15,
                    samples_path=Path(d)/'samples', status_path=Path(d)/'status',
                    read_memory=lambda: (8*1024**3, 4*1024**3))
                self.assertEqual(w.poll(), 'memory-paused')
                self.assertEqual(set(w.last['pids']) & {p.pid for p in groups}, {p.pid for p in groups})
                self.assertEqual(len(w.last['pids']), len(set(w.last['pids'])))
        finally:
            for process in groups:
                process.terminate()
            for process in groups:
                process.wait()

if __name__ == '__main__': unittest.main()
