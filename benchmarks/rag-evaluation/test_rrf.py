import unittest
from rrf import reciprocal_rank_fusion

class RrfTests(unittest.TestCase):
    def test_deduplicates_and_is_deterministic_on_ties(self):
        self.assertEqual(reciprocal_rank_fusion([["b", "a", "a"], ["a", "b"]], k=60),
                         [("a", 1 / 61 + 1 / 62), ("b", 1 / 61 + 1 / 62)])
    def test_limit_and_invalid_k(self):
        self.assertEqual(len(reciprocal_rank_fusion([["a", "b"]], limit=1)), 1)
        with self.assertRaises(ValueError): reciprocal_rank_fusion([["a"]], k=0)

if __name__ == "__main__": unittest.main()
