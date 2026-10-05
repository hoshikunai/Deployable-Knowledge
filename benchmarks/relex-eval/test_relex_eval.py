from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_set
import score
from common import mentioned, normalize, same_entity


def record(index: int, question_type: str = "compositional") -> dict:
    film, director = f"Film Number {index}", f"Jane Director {index}"
    return {
        "_id": f"q{index:03d}",
        "type": question_type,
        "question": f"Who directed {film}?",
        "answer": director,
        "context": [
            [film, [f"{film} is a 1999 drama film", f" directed by {director}."]],
            [director, [f"{director} was born in Lyon, France."]],
            ["Unrelated Page", ["Nothing to see here at all."]],
        ],
        "supporting_facts": [[film, 0], [director, 0]],
        "evidences": [[film, "director", director], [director, "place of birth", "Lyon"]],
    }


def prediction(question_id: str, relations: list[tuple[str, str, str, float]], entities: list[str]) -> dict:
    return {"id": question_id, "seconds": 0.5, "paragraphs": [{
        "title": "p",
        "entities": [{"text": text, "label": "other", "score": 0.9} for text in entities],
        "relations": [{"head": h, "relation": r, "tail": t, "score": s} for h, r, t, s in relations],
    }]}


class CommonTests(unittest.TestCase):
    def test_normalize_drops_case_punctuation_and_articles(self) -> None:
        self.assertEqual(normalize("The  Godfather, Part II!"), "godfather part ii")

    def test_exact_and_lenient_entity_matching(self) -> None:
        self.assertTrue(same_entity("Lyon", "lyon", lenient=False))
        self.assertFalse(same_entity("Lyon", "Lyon, France", lenient=False))
        self.assertTrue(same_entity("Lyon", "Lyon, France", lenient=True))
        self.assertFalse(same_entity("Lyon", "Paris", lenient=True))

    def test_mentioned_needs_whole_words(self) -> None:
        self.assertTrue(mentioned("Lyon", "born in Lyon, France"))
        self.assertFalse(mentioned("Lyo", "born in Lyon, France"))


class BuildTests(unittest.TestCase):
    def test_items_keep_gold_paragraphs_and_mark_findable_triples(self) -> None:
        items, dropped = build_set.load_items([record(1)])
        [item] = items
        self.assertEqual([paragraph["title"] for paragraph in item["paragraphs"]],
                         ["Film Number 1", "Jane Director 1"])
        self.assertEqual(item["paragraphs"][0]["text"], "Film Number 1 is a 1999 drama film directed by Jane Director 1.")
        self.assertTrue(all(gold["findable"] for gold in item["gold"]))
        self.assertEqual(dropped, {})

    def test_unwritten_entity_is_not_findable(self) -> None:
        raw = record(1)
        raw["evidences"].append(["Jane Director 1", "country of citizenship", "Germany"])
        [item], _ = build_set.load_items([raw])
        self.assertFalse(item["gold"][2]["findable"])

    def test_questions_without_evidences_are_dropped(self) -> None:
        raw = record(1)
        raw["evidences"] = []
        items, dropped = build_set.load_items([raw])
        self.assertEqual(items, [])
        self.assertEqual(dropped["no-evidences"], 1)

    def test_malformed_records_are_rejected(self) -> None:
        raw = record(1)
        raw["supporting_facts"].append(["Missing Title", 0])
        with self.assertRaisesRegex(build_set.SourceFormatError, "not in the context"):
            build_set.load_items([raw])
        broken = record(2)
        broken["evidences"] = [["only", "two"]]
        with self.assertRaisesRegex(build_set.SourceFormatError, "three strings"):
            build_set.load_items([broken])

    def test_sampling_is_stratified_and_repeatable(self) -> None:
        records = [record(index) for index in range(30)] + [record(100 + index, "comparison") for index in range(10)]
        items, _ = build_set.load_items(records)
        first = build_set.sample(items, 8, 42)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in build_set.sample(items, 8, 42)])
        self.assertEqual(sum(item["type"] == "comparison" for item in first), 2)

    def test_build_writes_relation_labels_by_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dev.json"
            source.write_text(json.dumps([record(index) for index in range(5)]), encoding="utf-8")
            target = Path(directory) / "set.json"
            result = build_set.build(source, 3, 42, ["person", "film"], target)
            self.assertEqual(len(result["items"]), 3)
            self.assertEqual(set(result["relationLabels"]), {"director", "place of birth"})
            self.assertEqual(result["goldTriples"], 6)
            self.assertTrue(target.is_file())


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        items, _ = build_set.load_items([record(1), record(2)])
        self.evaluation_set = {"items": items, "goldTriples": 4, "findableTriples": 4}

    def test_found_reversed_and_missing_outcomes(self) -> None:
        gold = self.evaluation_set["items"][0]["gold"][0]
        found = [("Film Number 1", "director", "Jane Director 1")]
        backwards = [("Jane Director 1", "director", "Film Number 1")]
        self.assertEqual(score.gold_outcome(gold, found, lenient=False), "found")
        self.assertEqual(score.gold_outcome(gold, backwards, lenient=False), "reversed")
        self.assertIsNone(score.gold_outcome(gold, [("Film Number 1", "producer", "Jane Director 1")], False))

    def test_thresholds_filter_predictions(self) -> None:
        predictions = {
            "q001": prediction("q001", [("Film Number 1", "director", "Jane Director 1", 0.95),
                                        ("Jane Director 1", "place of birth", "Lyon", 0.55)], []),
            "q002": prediction("q002", [], []),
        }
        rows = {row["threshold"]: row for row in score.evaluate(self.evaluation_set, predictions, False)["rows"]}
        self.assertEqual(rows[0.5]["recall"], 0.5)
        self.assertEqual(rows[0.7]["recall"], 0.25)
        self.assertEqual(rows[0.7]["precision"], 1.0)

    def test_misses_are_diagnosed(self) -> None:
        record_one = prediction("q001", [("Film Number 1", "director", "Jane Director 1", 0.6)],
                                ["Film Number 1", "Jane Director 1"])
        gold = self.evaluation_set["items"][0]["gold"]
        self.assertEqual(score.diagnose(gold[0], record_one, lenient=True), "below threshold")
        self.assertEqual(score.diagnose(gold[1], record_one, lenient=True), "entity missed")
        only_entities = prediction("q001", [], ["Film Number 1", "Jane Director 1"])
        self.assertEqual(score.diagnose(gold[0], only_entities, lenient=True), "relation missed")

    def test_report_renders_with_partial_predictions(self) -> None:
        predictions = {"q001": prediction("q001", [("Film Number 1", "director", "Jane Director 1", 0.9)],
                                          ["Film Number 1", "Jane Director 1"])}
        lines, summary = score.report(self.evaluation_set, predictions)
        self.assertEqual(summary["scoredQuestions"], 1)
        self.assertIn("1 of 2 questions scored", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
