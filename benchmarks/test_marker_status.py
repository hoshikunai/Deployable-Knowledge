import os, tempfile, unittest
from pathlib import Path
from benchmarks.benchmark_lease import scan_markers, validate_status, start_identity

class MarkerStatusTests(unittest.TestCase):
 def test_verified_and_malformed_markers_reported(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); run=root/'run'; run.mkdir(); (run/'benchmark-owner.json').write_text('{bad')
   self.assertFalse(scan_markers([root])[0]['verified'])
 def test_reused_pid_not_verified(self):
  self.assertFalse(scan_markers([])); self.assertIsNotNone(start_identity(os.getpid()))
 def test_taxonomy_schema(self):
  for status in ('completed','failed','timed-out','memory-paused','interrupted','blocked-active-run'):
   value={'status':status,'reason':'x','peakOwnedRss':0,'lastSample':{},'childIdentities':[],'checkpointCount':0,'resumeCommand':'resume'}
   if status=='completed': value.pop('resumeCommand')
   if status=='blocked-active-run': value['activeOwner']={}
   validate_status(value)
  with self.assertRaises(ValueError): validate_status({'status':'bogus'})

if __name__=='__main__': unittest.main()
