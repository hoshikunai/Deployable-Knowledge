from __future__ import annotations

import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import run as harness_run

from dataset_catalog import dataset_url, validate_dataset_name
from beir.datasets.data_loader import GenericDataLoader
from run import (
    canonicalize_qrels,
    canonicalize_rankings,
    atomic_write_json,
    record_query_checkpoint,
    parse_query_counts,
    register_document_mapping,
    select_queries,
    validate_resume_configuration,
    collapse_chunks_to_documents,
    load_query_checkpoints,
    run_query_searches,
    validate_public_protocol,
    validate_public_mapping,
    validate_public_results,
)
from app_client import DeployableKnowledgeClient
from run_suite import select_available_port


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

    def test_all_queries_preserve_dataset_order_without_sampling(self) -> None:
        selected, checkpoints, protocol = select_queries(
            self.queries, 0, "all", 42
        )
        self.assertEqual([query_id for query_id, _ in selected], list(self.queries))
        self.assertEqual(checkpoints, [20])
        self.assertEqual(protocol, "dataset-order")

    def test_public_protocol_rejects_wrong_counts_and_sampling(self) -> None:
        protocol = {"protocolVersion": "scifact-full-test-v1", "datasetName": "scifact",
                    "split": "test", "corpusDocumentCount": 5183, "queryCount": 300,
                    "sampling": "none", "resultDepth": 10,
                    "applicationSystems": ["bm25", "semantic", "hybrid"],
                    "metrics": {"cutoffs": [1, 3, 5, 10]}}
        with self.assertRaises(ValueError):
            validate_public_protocol(protocol, {"d": {}}, {"q": "x"}, {"q": {"d": 1}}, ["q"])

    def test_public_protocol_real_scifact_loader(self) -> None:
        root = Path(__file__).parent / "runs/public-comparable/scifact-full-test-v1"
        protocol = json.loads((root / "protocol.json").read_text())
        corpus, queries, qrels = GenericDataLoader(
            data_folder=str(Path(__file__).parent / "datasets/scifact")
        ).load(split="test")
        validate_public_protocol(protocol, corpus, queries, qrels, list(qrels))

    def test_public_protocol_methods_are_set_validated(self) -> None:
        protocol = {"protocolVersion": "scifact-full-test-v1", "datasetName": "scifact",
                    "split": "test", "corpusDocumentCount": 5183, "queryCount": 300,
                    "sampling": "none", "resultDepth": 10,
                    "applicationSystems": ["hybrid", "semantic", "bm25"],
                    "metrics": {"cutoffs": [1, 3, 5, 10]}}
        corpus = {f"d{i}": {} for i in range(5183)}
        queries = {f"q{i}": "x" for i in range(300)}
        qrels = {qid: {"d0": 1} for qid in queries}
        validate_public_protocol(protocol, corpus, queries, qrels, list(qrels))
        for systems in (["bm25", "semantic"], ["bm25", "semantic", "semantic"], ["bm25", "semantic", "other"]):
            with self.assertRaises(ValueError):
                validate_public_protocol(dict(protocol, applicationSystems=systems), corpus, queries, qrels, list(qrels))


    def test_public_result_guard_requires_exactly_ten(self) -> None:
        ranking = {f"d{i}": float(i) for i in range(10)}
        results = {method: {"q": ranking} for method in harness_run.METHODS}
        validate_public_results(results, ["q"], set(ranking), {"q": {"d0": 1}})
        with self.assertRaises(RuntimeError):
            validate_public_results({method: {"q": dict(list(ranking.items())[:9])} for method in harness_run.METHODS}, ["q"], set(ranking), {"q": {"d0": 1}})

    def test_public_mapping_guard_rejects_duplicate_mapping(self) -> None:
        mapping = {"beirToApplication": {"d": "a"}, "applicationToBeir": {"a": "d"},
                   "applicationToBeirAliases": {"a": ["d", "d2"]}, "skippedBeirDocuments": {},
                   "missingCorpusQrelDocumentIds": []}
        with self.assertRaises(RuntimeError):
            validate_public_mapping(mapping, {"d": {}}, {"q": {"d": 1}})

    def test_resume_checkpoint_skips_completed_query_and_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            record_query_checkpoint(path, "q1", {"semantic": {"d1": 1.0}}, 0.1)
            record_query_checkpoint(path, "q2", {"semantic": {}}, 0.2, "timeout")
            record_query_checkpoint(path, "q1", {"semantic": {"d2": 1.0}}, 0.3)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["queryId"] for record in records], ["q2", "q1"])
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())

    def test_resume_configuration_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_resume_configuration({"dataset": "scifact"}, {"dataset": "nfcorpus"})

    def test_malformed_markdown_base_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DeployableKnowledgeClient("[http://127.0.0.1:4179](http://127.0.0.1:4179)")

    def test_occupied_preferred_port_gets_an_available_port(self) -> None:
        from unittest.mock import patch
        with patch("run_suite.port_is_available", return_value=False), patch(
            "run_suite.socket.socket"
        ) as socket_factory:
            socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                "127.0.0.1",
                49152,
            )
            self.assertEqual(select_available_port(4179), 49152)

    def test_failure_checkpoint_keeps_empty_ranking_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            record_query_checkpoint(path, "failed", {"semantic": {}, "bm25": {}, "hybrid": {}}, 180.0, "timeout after retry")
            record = json.loads(path.read_text())
            self.assertEqual(record["queryId"], "failed")
            self.assertEqual(record["failure"], "timeout after retry")

    def test_search_and_ingestion_timeouts_are_separate(self) -> None:
        client = DeployableKnowledgeClient(
            "http://127.0.0.1:4179", search_timeout=180, ingestion_timeout=1800
        )
        self.assertEqual(client.search_timeout, 180)
        self.assertEqual(client.ingestion_timeout, 1800)


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


class FakeClient:
    def __init__(
        self,
        *,
        interrupt_on: str | None = None,
        fail_queries: set[str] | None = None,
    ) -> None:
        self.interrupt_on = interrupt_on
        self.fail_queries = fail_queries or set()
        self.ingest_calls: list[tuple[str, str]] = []
        self.search_calls: list[str] = []

    def heartbeat(self) -> None:
        return

    def ingest_document(self, title: str, text: str) -> dict[str, str]:
        self.ingest_calls.append((title, text))
        return {"documentId": "app-document"}

    def search(
        self, query: str, top_k: int
    ) -> dict[str, list[dict[str, object]]]:
        del top_k
        self.search_calls.append(query)
        if query == self.interrupt_on:
            raise KeyboardInterrupt("simulated interruption")
        if query in self.fail_queries:
            raise TimeoutError("simulated timeout")
        hits = [
            {
                "chunkId": f"chunk-{query}",
                "documentId": "app-document",
                "chunkIndex": 0,
            }
        ]
        return {method: hits for method in harness_run.METHODS}


class RunnerExecutionTests(unittest.TestCase):
    def arguments(self, *, resume: bool, run_name: str) -> Namespace:
        return Namespace(
            dataset="scifact",
            split="test",
            base_url="http://127.0.0.1:4179",
            search_depth=10,
            query_limit=0,
            query_counts="all",
            sample_seed=42,
            run_name=run_name,
            resume=resume,
            search_timeout=180,
            ingestion_timeout=1800,
            chunk_overfetch_factor=1,
            public_protocol=None,
            prepare_only=False,
        )

    def test_prepare_only_ingests_without_search(self) -> None:
        corpus = {"document": {"title": "Test document", "text": "one two three four five six"}}
        queries = {"q1": "query one"}
        qrels = {"q1": {"document": 1}}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(harness_run, "RUNS_ROOT", Path(directory) / "runs"),
                  patch.object(harness_run, "DATASETS_ROOT", Path(directory) / "datasets"),
                  patch.object(harness_run, "download_dataset", return_value=Path(directory) / "datasets/scifact"),
                  patch.object(harness_run, "GenericDataLoader", return_value=MagicMock(load=MagicMock(return_value=(corpus, queries, qrels)))),
                  patch.object(harness_run, "DeployableKnowledgeClient", return_value=client),
                  patch.object(harness_run, "parse_arguments", return_value=Namespace(**{**vars(self.arguments(resume=False, run_name="prepare")), "prepare_only": True}))):
                harness_run.main()
        self.assertEqual(len(client.ingest_calls), 1)
        self.assertEqual(client.search_calls, [])

    def test_main_resumes_interrupted_search_without_reingestion(self) -> None:
        corpus = {
            "document": {
                "title": "Test document",
                "text": "one two three four five six",
            }
        }
        queries = {"q1": "query one", "q2": "query two", "q3": "query three"}
        qrels = {query_id: {"document": 1} for query_id in queries}
        first_client = FakeClient(interrupt_on="query three")
        resumed_client = FakeClient()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            dataset_root = root / "datasets"
            loader = MagicMock()
            loader.load.return_value = (corpus, queries, qrels)

            with (
                patch.object(harness_run, "RUNS_ROOT", run_root),
                patch.object(harness_run, "DATASETS_ROOT", dataset_root),
                patch.object(
                    harness_run,
                    "download_dataset",
                    return_value=dataset_root / "scifact",
                ),
                patch.object(
                    harness_run,
                    "GenericDataLoader",
                    return_value=loader,
                ),
                patch.object(
                    harness_run,
                    "DeployableKnowledgeClient",
                    side_effect=[first_client, resumed_client],
                ),
                patch.object(
                    harness_run,
                    "parse_arguments",
                    side_effect=[
                        self.arguments(resume=False, run_name="resume-test"),
                        self.arguments(resume=True, run_name="resume-test"),
                    ],
                ),
                patch.object(
                    harness_run,
                    "evaluate_method",
                    return_value={"validated": True},
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    harness_run.main()

                run_directory = run_root / "scifact-resume-test"
                partial = load_query_checkpoints(
                    run_directory / "query-checkpoints.jsonl",
                    list(queries),
                )
                self.assertEqual(list(partial), ["q1", "q2"])
                self.assertFalse((run_directory / "metrics.json").exists())

                harness_run.main()

            self.assertEqual(len(first_client.ingest_calls), 1)
            self.assertEqual(resumed_client.ingest_calls, [])
            self.assertEqual(
                first_client.search_calls,
                ["query one", "query two", "query three"],
            )
            self.assertEqual(resumed_client.search_calls, ["query three"])

            run_directory = run_root / "scifact-resume-test"
            status = json.loads(
                (run_directory / "run-status.json").read_text()
            )
            rankings = json.loads(
                (run_directory / "document-rankings.json").read_text()
            )
            self.assertEqual(status["status"], "complete")
            self.assertEqual(list(rankings["hybrid"]), list(queries))
            self.assertTrue((run_directory / "metrics.json").is_file())

    def test_main_does_not_emit_metrics_for_unresolved_failures(self) -> None:
        corpus = {
            "document": {
                "title": "Test document",
                "text": "one two three four five six",
            }
        }
        queries = {"q1": "query one"}
        qrels = {"q1": {"document": 1}}
        client = FakeClient(fail_queries={"query one"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            dataset_root = root / "datasets"
            loader = MagicMock()
            loader.load.return_value = (corpus, queries, qrels)

            with (
                patch.object(harness_run, "RUNS_ROOT", run_root),
                patch.object(harness_run, "DATASETS_ROOT", dataset_root),
                patch.object(
                    harness_run,
                    "download_dataset",
                    return_value=dataset_root / "scifact",
                ),
                patch.object(
                    harness_run,
                    "GenericDataLoader",
                    return_value=loader,
                ),
                patch.object(
                    harness_run,
                    "DeployableKnowledgeClient",
                    return_value=client,
                ),
                patch.object(
                    harness_run,
                    "parse_arguments",
                    return_value=self.arguments(
                        resume=False, run_name="failed-test"
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    harness_run.main()

            run_directory = run_root / "scifact-failed-test"
            status = json.loads(
                (run_directory / "run-status.json").read_text()
            )
            failures = json.loads(
                (run_directory / "failures.json").read_text()
            )
            self.assertEqual(status["status"], "incomplete")
            self.assertEqual(len(failures), 1)
            self.assertFalse((run_directory / "metrics.json").exists())

    def test_resume_mismatch_fails_before_client_construction(self) -> None:
        corpus = {
            "document": {
                "title": "Test document",
                "text": "one two three four five six",
            }
        }
        queries = {"q1": "query one"}
        qrels = {"q1": {"document": 1}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            run_directory = run_root / "scifact-mismatch-test"
            run_directory.mkdir(parents=True)
            atomic_write_json(
                run_directory / "run-config.json",
                {"dataset": "nfcorpus"},
            )
            atomic_write_json(
                run_directory / "query-selection.json",
                {"queryIds": ["q1"]},
            )
            atomic_write_json(
                run_directory / "document-id-mapping.json",
                {},
            )
            dataset_root = root / "datasets"
            loader = MagicMock()
            loader.load.return_value = (corpus, queries, qrels)

            with (
                patch.object(harness_run, "RUNS_ROOT", run_root),
                patch.object(harness_run, "DATASETS_ROOT", dataset_root),
                patch.object(
                    harness_run,
                    "download_dataset",
                    return_value=dataset_root / "scifact",
                ),
                patch.object(
                    harness_run,
                    "GenericDataLoader",
                    return_value=loader,
                ),
                patch.object(
                    harness_run, "DeployableKnowledgeClient"
                ) as client_constructor,
                patch.object(
                    harness_run,
                    "parse_arguments",
                    return_value=self.arguments(
                        resume=True, run_name="mismatch-test"
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    harness_run.main()
                client_constructor.assert_not_called()

    def test_failed_checkpoint_is_retried_and_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "query-checkpoints.jsonl"
            status_path = Path(directory) / "run-status.json"
            selected_queries = [("q1", "query one")]
            failing_client = FakeClient(fail_queries={"query one"})

            _results, _raw, failures = run_query_searches(
                failing_client,
                selected_queries,
                {"app-document": "document"},
                10,
                checkpoint_path,
                status_path,
            )
            self.assertEqual(len(failures), 1)
            self.assertEqual(len(failing_client.search_calls), 2)

            resumed_client = FakeClient()
            results, _raw, failures = run_query_searches(
                resumed_client,
                selected_queries,
                {"app-document": "document"},
                10,
                checkpoint_path,
                status_path,
            )
            self.assertEqual(failures, [])
            self.assertEqual(resumed_client.search_calls, ["query one"])
            self.assertEqual(list(results["semantic"]), ["q1"])
            resumed_checkpoint = load_query_checkpoints(
                checkpoint_path, ["q1"]
            )["q1"]
            self.assertEqual(resumed_checkpoint["attemptCount"], 3)

    def test_malformed_and_duplicate_checkpoints_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "query-checkpoints.jsonl"
            checkpoint_path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                load_query_checkpoints(checkpoint_path, ["q1"])

            record = {
                "queryId": "q1",
                "rankings": {},
                "rawRanking": None,
                "elapsedSeconds": 1.0,
                "attemptCount": 1,
                "failure": "timeout",
            }
            checkpoint_path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                load_query_checkpoints(checkpoint_path, ["q1"])


class DuplicateDocumentTests(unittest.TestCase):
    def test_rankings_collapse_duplicate_ids_in_rank_order(self) -> None:
        normalized = canonicalize_rankings(
            {"query": {"MED-719": 3.0, "MED-724": 2.0}},
            {"MED-719": "application-id", "MED-724": "application-id"},
            {"application-id": "MED-719"},
        )
        self.assertEqual(normalized, {"query": {"MED-719": 3.0}})

    def test_chunk_collapse_keeps_first_highest_ranked_document_occurrence(self) -> None:
        collapsed = collapse_chunks_to_documents(
            [
                {"documentId": "app-1"},
                {"documentId": "app-1"},
                {"documentId": "app-2"},
            ],
            {"app-1": "DOC-1", "app-2": "DOC-2"},
            2,
        )
        self.assertEqual(list(collapsed), ["DOC-1", "DOC-2"])

    def test_rankings_reject_unknown_document_ids(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the ingested corpus"):
            canonicalize_rankings(
                {"query": {"unknown": 1.0}}, {}, {}
            )
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
