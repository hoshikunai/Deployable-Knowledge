from __future__ import annotations

import unittest

from build_openrag_dataset import align_sections, normalize_words, select_papers

INTRODUCTION = ("Introduction. Retrieval systems rank passages for every incoming question "
                "and the ranking quality decides what the reader model eventually sees")
METHOD = ("Method. Our pipeline splits each document into overlapping windows before "
          "embedding them with a compact bidirectional encoder trained on scientific text")
RESULTS = ("Results. Across seven benchmarks the hybrid configuration improves recall while "
           "keeping latency within the budget required for interactive desktop use")


class NormalizeWordsTests(unittest.TestCase):
    def test_drops_math_placeholders_and_short_words(self) -> None:
        words = normalize_words("We set $x = 2$ and ![table_0](table_0) the ﬁnal re-\nsult $$y$$")
        self.assertEqual(words, ["set", "and", "the", "final", "result"])


class AlignSectionsTests(unittest.TestCase):
    def test_sections_on_their_own_pages(self) -> None:
        ranges = align_sections([INTRODUCTION, METHOD, RESULTS],
                                [INTRODUCTION, METHOD, RESULTS])
        self.assertEqual(ranges, [(1, 2), (2, 3), (3, 3)])

    def test_section_starting_mid_page_and_running_to_the_last_page(self) -> None:
        pages = ["Title page with authors only", f"{INTRODUCTION} {METHOD[:60]}",
                 METHOD[60:], f"{RESULTS[:80]}", RESULTS[80:]]
        ranges = align_sections([INTRODUCTION, METHOD, RESULTS], pages)
        self.assertEqual(ranges, [(2, 2), (2, 4), (4, 5)])

    def test_unfound_section_is_none_and_does_not_move_the_floor(self) -> None:
        missing = "Appendix. Entirely different words never printed anywhere inside this paper"
        ranges = align_sections([INTRODUCTION, missing, METHOD], [INTRODUCTION, METHOD])
        self.assertEqual(ranges, [(1, 2), None, (2, 2)])

    def test_too_short_section_is_none(self) -> None:
        self.assertEqual(align_sections(["Short heading"], [INTRODUCTION]), [None])


class SelectPapersTests(unittest.TestCase):
    QRELS = {f"q{index}": {"doc_id": f"p{index % 4}", "section_id": 0} for index in range(8)}
    PAPERS = [f"p{index}" for index in range(10)]

    def test_selection_is_seeded_and_separates_roles(self) -> None:
        positives, negatives = select_papers(self.QRELS, self.PAPERS, 2, 3, seed=1)
        self.assertEqual((positives, negatives), select_papers(self.QRELS, self.PAPERS, 2, 3, seed=1))
        self.assertTrue(set(positives) <= {"p0", "p1", "p2", "p3"})
        self.assertFalse(set(negatives) & {"p0", "p1", "p2", "p3"})
        self.assertEqual((len(positives), len(negatives)), (2, 3))

    def test_none_takes_every_paper(self) -> None:
        positives, negatives = select_papers(self.QRELS, self.PAPERS, None, None, seed=1)
        self.assertEqual(len(positives) + len(negatives), 10)

    def test_asking_for_too_many_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            select_papers(self.QRELS, self.PAPERS, 5, 0, seed=1)


if __name__ == "__main__":
    unittest.main()
