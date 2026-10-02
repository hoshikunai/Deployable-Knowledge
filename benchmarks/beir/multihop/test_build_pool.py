from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_pool
from build_pool import SourceFormatError


def sentences(title: str, filler: str = "is a subject described in several plain words") -> list[str]:
    return [f"{title} {filler}.", " It appears in this synthetic fixture."]


def hotpot_record(index: int, question_type: str = "bridge") -> dict:
    first, second = f"Alpha Topic {index}", f"Beta Topic {index}"
    question = (f"What links Alpha Topic {index} to its partner?" if question_type == "bridge"
                else f"Is Alpha Topic {index} older than Beta Topic {index}?")
    return {
        "_id": f"q{index:04d}",
        "question": question,
        "answer": "yes",
        "type": question_type,
        "level": "hard",
        "supporting_facts": [[first, 0], [second, 0]],
        "context": [
            [first, sentences(first)],
            [second, sentences(second)],
            [f"Gamma Topic {index}", sentences(f"Gamma Topic {index}")],
            ["Shared Distractor", sentences("Shared Distractor")],
        ],
    }


def hotpot_records(bridge: int = 30, comparison: int = 10) -> list[dict]:
    return ([hotpot_record(index) for index in range(bridge)]
            + [hotpot_record(bridge + index, "comparison") for index in range(comparison)])


def musique_record(index: int, answerable: bool = True) -> dict:
    return {
        "id": f"2hop__{index}_{index + 1}",
        "question": f"Who founded the company that makes Widget {index}?",
        "answer": "Somebody",
        "answer_aliases": ["Some Body"],
        "answerable": answerable,
        "paragraphs": [
            {"idx": 0, "title": f"Founder {index}", "is_supporting": True,
             "paragraph_text": f"Founder {index} started the maker company long ago in a city."},
            {"idx": 1, "title": f"Widget {index}", "is_supporting": True,
             "paragraph_text": f"Widget {index} is made by the maker company in large numbers."},
            {"idx": 2, "title": f"Unrelated {index}", "is_supporting": False,
             "paragraph_text": f"Unrelated {index} is a paragraph about something else entirely."},
        ],
        "question_decomposition": [
            {"id": 1, "question": f"Widget {index} >> manufacturer", "answer": "maker",
             "paragraph_support_idx": 1},
            {"id": 2, "question": "#1 >> founded by", "answer": "Somebody", "paragraph_support_idx": 0},
        ],
    }


def wiki2_record(index: int) -> dict:
    record = hotpot_record(index, "compositional")
    del record["level"]
    record["evidences"] = [[f"Alpha Topic {index}", "partner", f"Beta Topic {index}"]]
    return record


class BuildTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.datasets = self.root / "datasets"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def source(self, records: list[dict], name: str = "source.json", jsonl: bool = False) -> Path:
        path = self.root / name
        if jsonl:
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        else:
            path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def build(self, records: list[dict], dataset: str = "hotpotqa", name: str = "multihop-test",
              per_split: int = 10, **options) -> dict:
        source = self.source(records, jsonl=dataset == "musique")
        return build_pool.build(dataset, source, name, per_split, options.pop("seed", 42),
                                options.pop("filler_count", 0), options.pop("filler_source", None),
                                options.pop("force", False), datasets_root=self.datasets)

    def qrels(self, name: str, split: str) -> list[dict[str, str]]:
        with (self.datasets / name / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def corpus(self, name: str) -> list[dict]:
        text = (self.datasets / name / "corpus.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]


class SamplingTests(unittest.TestCase):
    def test_splits_are_disjoint_stratified_and_repeatable(self) -> None:
        loaded = build_pool.load_hotpotqa(hotpot_records())
        first = build_pool.sample_splits(loaded.questions, 10, 42)
        second = build_pool.sample_splits(loaded.questions, 10, 42)
        dev_ids = [question.id for question in first["dev"]]
        test_ids = [question.id for question in first["test"]]
        self.assertEqual(len(dev_ids), 10)
        self.assertEqual(len(test_ids), 10)
        self.assertFalse(set(dev_ids) & set(test_ids))
        self.assertEqual(dev_ids, [question.id for question in second["dev"]])
        for split in ("dev", "test"):
            types = [question.type for question in first[split]]
            self.assertIn(types.count("comparison"), (2, 3))

    def test_too_few_questions_is_rejected(self) -> None:
        loaded = build_pool.load_hotpotqa(hotpot_records(5, 2))
        with self.assertRaisesRegex(ValueError, "eligible questions"):
            build_pool.sample_splits(loaded.questions, 10, 42)

    def test_quotas_sum_to_total(self) -> None:
        self.assertEqual(sum(build_pool.quotas({"a": 7, "b": 3, "c": 1}, 5).values()), 5)


class HopRoleTests(unittest.TestCase):
    def test_named_title_is_first_hop_and_other_is_second(self) -> None:
        hops = build_pool.heuristic_hops("Where was Alpha Band formed?", ["Alpha Band", "Beta City"])
        self.assertEqual(hops, {"Alpha Band": 1, "Beta City": 2})

    def test_parenthetical_disambiguation_is_ignored(self) -> None:
        self.assertTrue(build_pool.mentions_title("Who sang in Paris?", "Paris (band)"))

    def test_no_named_title_leaves_hops_unknown(self) -> None:
        hops = build_pool.heuristic_hops("Who did it?", ["Alpha Band", "Beta City"])
        self.assertEqual(hops, {"Alpha Band": None, "Beta City": None})

    def test_musique_hops_follow_the_decomposition(self) -> None:
        loaded = build_pool.load_musique([musique_record(1), musique_record(2, answerable=False)])
        self.assertEqual(loaded.dropped["unanswerable"], 1)
        [question] = loaded.questions
        self.assertEqual(question.type, "2-hop")
        self.assertEqual({paragraph.title: paragraph.hop for paragraph in question.gold},
                         {"Founder 1": 2, "Widget 1": 1})
        self.assertEqual(question.answers, ("Somebody", "Some Body"))


class ValidationTests(unittest.TestCase):
    def test_missing_key_is_rejected(self) -> None:
        record = hotpot_record(1)
        del record["context"]
        with self.assertRaisesRegex(SourceFormatError, "missing 'context'"):
            build_pool.load_hotpotqa([record])

    def test_supporting_title_outside_context_is_rejected(self) -> None:
        record = hotpot_record(1)
        record["supporting_facts"].append(["Nowhere", 0])
        with self.assertRaisesRegex(SourceFormatError, "not in the context"):
            build_pool.load_hotpotqa([record])

    def test_wrong_type_is_rejected(self) -> None:
        record = musique_record(1)
        record["answerable"] = "yes"
        with self.assertRaisesRegex(SourceFormatError, "answerable"):
            build_pool.load_musique([record])

    def test_short_gold_paragraph_drops_the_question(self) -> None:
        record = hotpot_record(1)
        record["context"][0][1] = ["Tiny."]
        loaded = build_pool.load_hotpotqa([record, hotpot_record(2)])
        dropped = loaded.dropped.copy()
        eligible = build_pool.eligible_questions(loaded.questions, dropped)
        self.assertEqual([question.id for question in eligible], ["q0002"])
        self.assertEqual(dropped["short-gold-paragraph"], 1)

    def test_names_must_carry_the_multihop_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start with 'multihop-'"):
            build_pool.validate_name("hotpotqa-pool")
        self.assertEqual(build_pool.validate_name("multihop-hotpotqa"), "multihop-hotpotqa")

    def test_paragraph_ids_are_stable_and_title_sensitive(self) -> None:
        first = build_pool.paragraph_id("hp", "Title", "Text body")
        self.assertEqual(first, build_pool.paragraph_id("hp", "Title", "Text body"))
        self.assertNotEqual(first, build_pool.paragraph_id("hp", "Other", "Text body"))
        self.assertTrue(first.startswith("hp-"))


class BuildTests(BuildTestCase):
    def test_build_writes_a_beir_layout_with_deduplicated_corpus(self) -> None:
        manifest = self.build(hotpot_records())
        target = self.datasets / "multihop-test"
        corpus = self.corpus("multihop-test")
        corpus_ids = {document["_id"] for document in corpus}
        self.assertEqual(len(corpus_ids), len(corpus))
        self.assertEqual(sum(document["title"] == "Shared Distractor" for document in corpus), 1)
        for split in ("dev", "test"):
            rows = self.qrels("multihop-test", split)
            self.assertEqual(len({row["query-id"] for row in rows}), 10)
            self.assertEqual(len(rows), 20)
            self.assertTrue({row["corpus-id"] for row in rows} <= corpus_ids)
        sidecar = json.loads((target / "multihop.json").read_text(encoding="utf-8"))
        self.assertEqual(len(sidecar["queries"]), 20)
        entry = next(iter(sidecar["queries"].values()))
        self.assertEqual({gold["hop"] for gold in entry["gold"]} - {None}, {1, 2} if entry["type"] == "bridge"
                         else {1})
        self.assertEqual(manifest["kind"], "multi-hop")
        self.assertEqual(manifest["corpus"]["paragraphs"], len(corpus))
        self.assertFalse(set(manifest["splits"]["dev"]) & set(manifest["splits"]["test"]))
        self.assertFalse((target / "evidence-triples.json").exists())
        self.assertEqual([path.name for path in self.datasets.iterdir()], ["multihop-test"])

    def test_existing_dataset_is_not_replaced_without_force(self) -> None:
        self.build(hotpot_records())
        with self.assertRaises(FileExistsError):
            self.build(hotpot_records())
        self.build(hotpot_records(), force=True)

    def test_musique_build_keeps_hops(self) -> None:
        records = [musique_record(index) for index in range(0, 40, 2)]
        manifest = self.build(records, dataset="musique", per_split=5)
        self.assertEqual(manifest["hopRoles"], {"known": 20, "unknown": 0})
        self.assertTrue(all(document["_id"].startswith("mq-") for document in self.corpus("multihop-test")))

    def test_2wiki_build_writes_evidence_triples(self) -> None:
        self.build([wiki2_record(index) for index in range(20)], dataset="2wiki", per_split=5)
        triples = json.loads((self.datasets / "multihop-test" / "evidence-triples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(triples), 10)
        self.assertEqual(len(next(iter(triples.values()))[0]), 3)

    def filler_source(self) -> Path:
        record = {
            **hotpot_record(900),
            "context": [
                ["Fresh Filler One", sentences("Fresh Filler One")],
                ["Fresh Filler Two", sentences("Fresh Filler Two")],
                ["Shared Distractor", sentences("Shared Distractor", "has different words this time around")],
                ["Partner", sentences("Partner")],
            ],
            "supporting_facts": [["Fresh Filler One", 0], ["Fresh Filler Two", 0]],
        }
        return self.source([record], name="train.json")

    def test_filler_excludes_pooled_and_mentioned_titles(self) -> None:
        manifest = self.build(hotpot_records(), filler_count=2, filler_source=self.filler_source())
        self.assertEqual(manifest["corpus"]["fillerParagraphs"], 2)
        titles = [document["title"] for document in self.corpus("multihop-test")]
        self.assertIn("Fresh Filler One", titles)
        self.assertIn("Fresh Filler Two", titles)
        self.assertEqual(titles.count("Shared Distractor"), 1)
        self.assertNotIn("Partner", titles)

    def test_filler_request_beyond_supply_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "filler paragraphs are available"):
            self.build(hotpot_records(), filler_count=3, filler_source=self.filler_source())
        self.assertFalse((self.datasets / "multihop-test").exists())

    def test_filler_requires_both_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "given together"):
            self.build(hotpot_records(), filler_count=5)


if __name__ == "__main__":
    unittest.main()
