from __future__ import annotations

import unittest

from dataset_catalog import dataset_url, validate_dataset_name
from run import (
    canonicalize_qrels,
    parse_query_counts,
    register_document_mapping,
    select_queries,
)


class QuerySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = {
            str(index): f"query {index}"
            for index in range(1, 21)
        }

    def test_query_counts_are_sorted_deduplicated_and_resolve_all(self) -> None:
        self.assertEqual(
            parse_query_counts("10,5,10,all", available=20),
            [5, 10, 20],
        )

    def test_query_counts_reject_values_larger_than_the_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "only has 20"):
            parse_query_counts("21", available=20)

    def test_seeded_samples_are_repeatable_and_nested(self) -> None:
        first, checkpoints, protocol = select_queries(
            self.queries,
            query_limit=0,
            query_counts="5,10",
            sample_seed=42,
        )
        second, _, _ = select_queries(
            self.queries,
            query_limit=0,
            query_counts="5,10",
            sample_seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(checkpoints, [5, 10])
        self.assertEqual(protocol, "seeded-shuffle")
        self.assertEqual(first[:5], second[:5])

    def test_query_limit_preserves_dataset_order(self) -> None:
        selected, checkpoints, protocol = select_queries(
            self.queries,
            query_limit=3,
            query_counts=None,
            sample_seed=42,
        )

        self.assertEqual([query_id for query_id, _ in selected], ["1", "2", "3"])
        self.assertEqual(checkpoints, [3])
        self.assertEqual(protocol, "dataset-order")


class DatasetCatalogTests(unittest.TestCase):
    def test_dataset_url_uses_the_official_archive_pattern(self) -> None:
        self.assertEqual(
            dataset_url("nfcorpus"),
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "thakur/BEIR/datasets/nfcorpus.zip",
        )

    def test_dataset_names_reject_path_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid BEIR dataset"):
            validate_dataset_name("../nfcorpus")


class DuplicateDocumentTests(unittest.TestCase):
    def test_duplicate_beir_documents_share_a_canonical_id(self) -> None:
        beir_to_application: dict[str, str] = {}
        application_to_beir: dict[str, str] = {}
        aliases: dict[str, list[str]] = {}

        register_document_mapping(
            "MED-719",
            "application-id",
            beir_to_application,
            application_to_beir,
            aliases,
        )
        register_document_mapping(
            "MED-724",
            "application-id",
            beir_to_application,
            application_to_beir,
            aliases,
        )

        self.assertEqual(
            beir_to_application,
            {
                "MED-719": "application-id",
                "MED-724": "application-id",
            },
        )
        self.assertEqual(
            application_to_beir,
            {"application-id": "MED-719"},
        )
        self.assertEqual(
            aliases,
            {"application-id": ["MED-719", "MED-724"]},
        )

    def test_qrels_for_duplicates_are_merged(self) -> None:
        normalized = canonicalize_qrels(
            qrels={"query": {"MED-719": 1, "MED-724": 2}},
            beir_to_application={
                "MED-719": "application-id",
                "MED-724": "application-id",
            },
            application_to_beir={"application-id": "MED-719"},
        )

        self.assertEqual(normalized, {"query": {"MED-719": 2}})

    def test_qrels_for_known_missing_corpus_documents_are_preserved(
        self,
    ) -> None:
        normalized = canonicalize_qrels(
            qrels={"query": {"missing-document": 1}},
            beir_to_application={},
            application_to_beir={},
            missing_corpus_document_ids={"missing-document"},
        )

        self.assertEqual(normalized, {"query": {"missing-document": 1}})

    def test_qrels_for_unexpected_uningested_documents_fail(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Qrels reference a document that was not ingested",
        ):
            canonicalize_qrels(
                qrels={"query": {"unexpected-document": 1}},
                beir_to_application={},
                application_to_beir={},
            )


if __name__ == "__main__":
    unittest.main()
