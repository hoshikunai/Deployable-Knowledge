import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).with_name('model_loader.py')
spec = importlib.util.spec_from_file_location('hakari_loader', MODULE)
loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)

class ProtocolTests(unittest.TestCase):
    def test_loopback_only(self):
        with self.assertRaises(ValueError): loader.LocalTypeScriptReranker('https://example.com/rerank')

    def test_response_shape_and_aliases(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return json.dumps({'ranked': [{'id': '0', 'score': .9, 'rank': 1}]}).encode()
        with patch.object(loader.request, 'urlopen', return_value=Response()):
            model = loader.LocalTypeScriptReranker()
            self.assertEqual(model.predict([['q', 'x']])[0], 0.9)
            self.assertIs(model.predict.__func__, model.__call__.__func__)

if __name__ == '__main__': unittest.main()
