from __future__ import annotations

import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import run as harness_run

from dataset_catalog import dataset_url, load_pdf_manifest, split_ranking_unit, validate_dataset_name
from beir.datasets.data_loader import GenericDataLoader
from run import (
    build_query_result,
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
from run_suite import parsing_environment, select_available_port, validate_existing_runtime_schema


class DocumentDepthTests(unittest.TestCase):
    def test_build_query_result_retains_first_100_unique_documents(self) -> None:
        hits = [
            {"chunkId": f"c{index}", "documentId": f"app{index}", "chunkIndex": 0}
            for index in range(120)
        ]
        response = {method: hits for method in harness_run.METHODS}
        mapping = {f"app{index}": f"doc{index}" for index in range(120)}
        rankings, _raw = build_query_result("q", "query", response, mapping, 100)
        for ranking in rankings.values():
            self.assertEqual(len(ranking), 100)
            self.assertEqual(list(ranking)[-1], "doc99")

    def test_100_document_evaluation_includes_recall_at_100(self) -> None:
        rankings = {"q": {f"d{index}": float(100 - index) for index in range(100)}}
        metrics = harness_run.evaluate_method(
            {"q": {"d99": 1}}, rankings, [1, 3, 5, 10, 100],
        )
        self.assertEqual(metrics["recall"]["Recall@100"], 1.0)


    def test_mrr_ignores_the_query_document_like_other_metrics(self) -> None:
        # ArguAna-style: the query's own document ranks first but is not a relevant answer.
        rankings = {"q1": {"q1": 3.0, "d2": 2.0, "d3": 1.0}}
        metrics = harness_run.evaluate_method({"q1": {"d2": 1}}, rankings, [1, 3])
        self.assertEqual(metrics["mrr"]["MRR@1"], 1.0)
        self.assertEqual(metrics["mrr"]["MRR@3"], 1.0)
        self.assertEqual(metrics["mrr"]["MRR@3"], metrics["map"]["MAP@3"])
        self.assertIn("q1", rankings["q1"])  # the caller's rankings are not modified


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
            checkpoints: dict = {}
            record_query_checkpoint(path, checkpoints, "q1", {"semantic": {"d1": 1.0}}, 0.1)
            record_query_checkpoint(path, checkpoints, "q2", {"semantic": {}}, 0.2, "timeout")
            record_query_checkpoint(path, checkpoints, "q1", {"semantic": {"d2": 1.0}}, 0.3)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["queryId"] for record in records], ["q2", "q1"])
            self.assertEqual(records[1]["rankings"], {"semantic": {"d2": 1.0}})
            self.assertEqual(list(checkpoints), ["q2", "q1"])
            self.assertEqual([path.name], [entry.name for entry in Path(directory).iterdir()])

    def test_checkpoint_appends_new_queries_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            checkpoints: dict = {}
            record_query_checkpoint(path, checkpoints, "q1", {"semantic": {}}, 0.1)
            inode = path.stat().st_ino
            record_query_checkpoint(path, checkpoints, "q2", {"semantic": {}}, 0.2)
            self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_checkpoint_cut_off_mid_append_is_ignored_then_compacted(self) -> None:
        complete = {
            "queryId": "q1", "rankings": {m: {"d": 1.0} for m in ("bm25", "semantic", "hybrid")},
            "rawRanking": {"methods": {m: [] for m in ("bm25", "semantic", "hybrid")}},
            "elapsedSeconds": 1.0, "attemptCount": 1, "failure": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(json.dumps(complete) + "\n" + '{"queryId": "q2", "rank', encoding="utf-8")
            checkpoints = load_query_checkpoints(path, ["q1", "q2"])
            self.assertEqual(list(checkpoints), ["q1"])

            record_query_checkpoint(path, checkpoints, "q2", {"semantic": {}}, 0.2, "timeout")
            self.assertTrue(path.read_text().endswith("\n"))
            self.assertEqual(list(load_query_checkpoints(path, ["q1", "q2"])), ["q1", "q2"])

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
            record_query_checkpoint(path, {}, "failed", {"semantic": {}, "bm25": {}, "hybrid": {}}, 180.0, "timeout after retry")
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

    def test_split_ranking_unit_follows_the_manifest(self) -> None:
        manifest = {"splitUnits": {"test": "page", "doc": "document"}}
        self.assertEqual(split_ranking_unit(None, "test"), "document")
        self.assertEqual(split_ranking_unit({}, "test"), "page")
        self.assertEqual(split_ranking_unit(manifest, "test"), "page")
        self.assertEqual(split_ranking_unit(manifest, "doc"), "document")

    def test_pdf_manifest_rejects_unknown_split_units(self) -> None:
        documents = [{"id": "paper", "file": "pdfs/paper.pdf", "pages": 3}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pdf-manifest.json"
            path.write_text(json.dumps({"documents": documents,
                                        "splitUnits": {"doc": "document"}}))
            self.assertEqual(load_pdf_manifest(Path(directory))["splitUnits"], {"doc": "document"})
            path.write_text(json.dumps({"documents": documents, "splitUnits": {"doc": "section"}}))
            with self.assertRaisesRegex(RuntimeError, "splitUnits"):
                load_pdf_manifest(Path(directory))


class ExistingRuntimeSchemaTests(unittest.TestCase):
    def test_watchdog_artifacts_are_separate_from_fresh_run_directory(self) -> None:
        from run_suite import MemoryWatchdog
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'arguana-run'; supervisor=root/'.supervisor-arguana-run'
            watchdog=MemoryWatchdog([999999], interval=0, status_path=supervisor/'status.json', samples_path=supervisor/'memory-samples.jsonl')
            watchdog.poll()
            self.assertFalse(run.exists())
            self.assertTrue(supervisor.exists())
            self.assertEqual(supervisor, root/'.supervisor-arguana-run')
    def test_rejects_marker_mismatch(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory); (runtime / "beir-dataset.json").write_text('{"dataset":"fiqa"}')
            with sqlite3.connect(runtime / "app.db") as db:
                for table in ("__drizzle_migrations", "app_state", "documents", "document_chunks"):
                    db.execute(f"CREATE TABLE {table} (id TEXT)")
                db.execute("INSERT INTO __drizzle_migrations VALUES ('x')")
            with self.assertRaisesRegex(RuntimeError, "marker mismatch"):
                validate_existing_runtime_schema(runtime, dataset="arguana")

    def test_rejects_missing_migration(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory); (runtime / "beir-dataset.json").write_text('{"dataset":"arguana"}')
            with sqlite3.connect(runtime / "app.db") as db:
                for table in ("__drizzle_migrations", "app_state", "documents", "document_chunks"):
                    db.execute(f"CREATE TABLE {table} (id TEXT)")
            with self.assertRaisesRegex(RuntimeError, "no applied migration"):
                validate_existing_runtime_schema(runtime, dataset="arguana")

    def test_rejects_document_id_mapping_mismatch(self) -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            runtime=Path(directory); (runtime/"beir-dataset.json").write_text('{"dataset":"arguana"}')
            with sqlite3.connect(runtime/"app.db") as db:
                for table in ("__drizzle_migrations","app_state","document_chunks"): db.execute(f"CREATE TABLE {table} (id TEXT)")
                db.execute("CREATE TABLE documents (id TEXT)"); db.execute("INSERT INTO __drizzle_migrations VALUES ('x')"); db.execute("INSERT INTO documents VALUES ('actual')")
            mapping=runtime/"mapping.json"; mapping.write_text(json.dumps({"applicationToBeir":{"expected":"document"}}))
            with self.assertRaisesRegex(RuntimeError,"document IDs"):
                validate_existing_runtime_schema(runtime,dataset="arguana",mapping=mapping)
    def test_accepts_migrated_existing_runtime_without_writing(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            database_path = runtime / "app.db"
            with sqlite3.connect(database_path) as database:
                for table in (
                    "__drizzle_migrations", "app_state", "documents", "document_chunks"
                ):
                    database.execute(f"CREATE TABLE {table} (id TEXT)")
                database.execute("INSERT INTO __drizzle_migrations VALUES ('applied')")

            before = database_path.stat().st_mtime_ns
            validate_existing_runtime_schema(runtime)
            self.assertEqual(database_path.stat().st_mtime_ns, before)

    def test_rejects_incomplete_runtime(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with sqlite3.connect(runtime / "app.db") as database:
                database.execute("CREATE TABLE app_state (id TEXT)")
            with self.assertRaisesRegex(RuntimeError, "missing required tables"):
                validate_existing_runtime_schema(runtime)


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


class DeviceClient(FakeClient):
    def __init__(self, *, vulkan: list[dict[str, object]] | None = None,
                 backend: str = "cpu") -> None:
        super().__init__()
        self.vulkan = vulkan or []
        self.backend = backend
        self.device = "auto"
        self.events: list[str] = []

    def vulkan_devices(self) -> list[dict[str, object]]:
        return self.vulkan

    def set_embedding_device(self, device: str) -> dict[str, object]:
        self.events.append(f"device:{device}")
        self.device = device
        return {"settings": {"device": device}}

    def embedding_status(self) -> dict[str, object]:
        return {"settings": {"device": self.device}, "backend": self.backend}

    def ingest_document(self, title: str, text: str) -> dict[str, str]:
        self.events.append("ingest")
        return super().ingest_document(title, text)


class SharedCacheLinkTests(unittest.TestCase):
    def test_windows_links_shared_caches_with_junctions_once(self) -> None:
        import sys
        import types
        import run_suite

        created: list[tuple[str, str]] = []

        def create_junction(target: str, link: str) -> None:
            created.append((target, link))
            Path(link).symlink_to(target, target_is_directory=True)

        fake_winapi = types.SimpleNamespace(CreateJunction=create_junction)
        with tempfile.TemporaryDirectory() as root:
            shared = Path(root) / "shared-tables"
            shared.mkdir()
            with (patch.object(run_suite, "RUNTIME_ROOT", Path(root) / "runtimes"),
                  patch.object(run_suite, "SHARED_MODEL_CACHE", Path(root) / "missing"),
                  patch.object(run_suite, "SHARED_EMBEDDING_CACHE", Path(root) / "missing"),
                  patch.object(run_suite, "SHARED_OCR_CACHE", Path(root) / "missing"),
                  patch.object(run_suite, "SHARED_TABLE_CACHE", shared),
                  patch.object(run_suite, "WINDOWS", True),
                  patch.dict(sys.modules, {"_winapi": fake_winapi})):
                runtime = run_suite.prepare_runtime("scifact", share_model_cache=True)
                run_suite.prepare_runtime("scifact", share_model_cache=True)
            link = runtime / ".cache" / "shared-tables"
            self.assertEqual(created, [(str(shared.resolve()), str(link))])
            self.assertTrue(run_suite.is_directory_link(link))


class EmbeddingDeviceTests(unittest.TestCase):
    def test_auto_leaves_the_app_default_alone(self) -> None:
        client = DeviceClient(backend="vulkan")
        harness_run.apply_embedding_device(client, "auto")
        harness_run.verify_embedding_device(client, "auto", "probe")
        self.assertEqual(client.events, [])
        self.assertEqual(client.search_calls, [])

    def test_vulkan_needs_a_visible_device(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No Vulkan device"):
            harness_run.apply_embedding_device(DeviceClient(), "vulkan")
        gpus = [{"index": 0, "name": "AMD Radeon 780M"}, {"index": 1, "name": "AMD Radeon RX 7700S"}]
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            harness_run.apply_embedding_device(DeviceClient(vulkan=gpus), "vulkan", 2)
        client = DeviceClient(vulkan=gpus)
        harness_run.apply_embedding_device(client, "vulkan", 1)
        self.assertEqual(client.events, ["device:gpu"])

    def test_verification_rejects_the_wrong_device_or_backend(self) -> None:
        client = DeviceClient(backend="vulkan")
        with self.assertRaisesRegex(RuntimeError, "embeds on 'auto'"):
            harness_run.verify_embedding_device(client, "cpu", "probe")
        client.device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "'vulkan' backend"):
            harness_run.verify_embedding_device(client, "cpu", "probe")
        client.backend = "cpu"
        harness_run.verify_embedding_device(client, "cpu", "probe")
        self.assertEqual(client.search_calls, ["probe", "probe", "probe"])

    def test_prepare_sets_the_device_before_ingesting_and_records_it(self) -> None:
        corpus = {"document": {"title": "Test document", "text": "one two three four five six"}}
        queries = {"q1": "query one"}
        qrels = {"q1": {"document": 1}}
        client = DeviceClient(backend="cpu")
        arguments = Namespace(
            dataset="scifact", split="test", base_url="http://127.0.0.1:4179", search_depth=10,
            query_limit=0, query_counts="all", sample_seed=42, run_name="prepare", resume=False,
            search_timeout=180, ingestion_timeout=1800, chunk_overfetch_factor=1,
            public_protocol=None, prepare_only=True, embedding_device="cpu", vulkan_device=-1,
        )
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            with (patch.object(harness_run, "RUNS_ROOT", runs),
                  patch.object(harness_run, "DATASETS_ROOT", Path(directory) / "datasets"),
                  patch.object(harness_run, "download_dataset", return_value=Path(directory) / "datasets/scifact"),
                  patch.object(harness_run, "GenericDataLoader", return_value=MagicMock(load=MagicMock(return_value=(corpus, queries, qrels)))),
                  patch.object(harness_run, "DeployableKnowledgeClient", return_value=client),
                  patch.object(harness_run, "parse_arguments", return_value=arguments)):
                harness_run.main()
            run_directory = runs / "scifact-prepare"
            mapping = json.loads((run_directory / "document-id-mapping.json").read_text())
            config = json.loads((run_directory / "run-config.json").read_text())
        self.assertEqual(client.events, ["device:cpu", "ingest"])
        self.assertEqual(mapping["embeddingDevice"], "cpu")
        self.assertEqual(config["embeddingDevice"], "cpu")
        self.assertNotIn("vulkanDevice", config)


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

    def test_main_reuses_existing_mapping_without_ingestion(self) -> None:
        corpus = {"document": {"title": "Test document", "text": "one two three"}}
        queries = {"q1": "query one"}; qrels = {"q1": {"document": 1}}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "mapping.json"
            source.write_text(json.dumps({"beirToApplication": {"document": "app-document"},
                "applicationToBeir": {"app-document": "document"},
                "applicationToBeirAliases": {}, "missingCorpusQrelDocumentIds": [],
                "skippedBeirDocuments": {}}))
            args = {**vars(self.arguments(resume=False, run_name="reuse")),
                    "prepare_only": True, "reuse_existing_runtime": True,
                    "reuse_document_mapping": source, "runtime_fingerprint": "fixture"}
            with (patch.object(harness_run, "RUNS_ROOT", root / "runs"),
                  patch.object(harness_run, "DATASETS_ROOT", root / "datasets"),
                  patch.object(harness_run, "download_dataset", return_value=root / "datasets/scifact"),
                  patch.object(harness_run, "GenericDataLoader", return_value=MagicMock(load=MagicMock(return_value=(corpus, queries, qrels)))),
                  patch.object(harness_run, "DeployableKnowledgeClient", return_value=client),
                  patch.object(harness_run, "parse_arguments", return_value=Namespace(**args))):
                harness_run.main()
            config = json.loads((root / "runs/scifact-reuse/run-config.json").read_text())
            self.assertEqual(client.ingest_calls, []); self.assertEqual(client.search_calls, [])
            self.assertTrue((root / "runs/scifact-reuse/document-id-mapping.json").is_file())
            for key in ("reuseExistingRuntime", "mappingSha256", "runtimeFingerprint", "corpusFingerprint"):
                self.assertIn(key, config)

    def test_reuse_resume_rejects_changed_mapping_before_client(self) -> None:
        corpus = {"document": {"title": "Test", "text": "one two three"}}
        q = {"q1": "query"}; qr = {"q1": {"document": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "mapping.json"
            mapping = {"beirToApplication":{"document":"app-document"},"applicationToBeir":{"app-document":"document"},"applicationToBeirAliases":{},"missingCorpusQrelDocumentIds":[],"skippedBeirDocuments":{}}
            source.write_text(json.dumps(mapping)); client = FakeClient()
            base = {**vars(self.arguments(resume=False, run_name="reuse")),"prepare_only":True,"reuse_existing_runtime":True,"reuse_document_mapping":source,"runtime_fingerprint":"fixture"}
            common = [patch.object(harness_run,"RUNS_ROOT",root/"runs"),patch.object(harness_run,"DATASETS_ROOT",root/"datasets"),patch.object(harness_run,"download_dataset",return_value=root/"datasets/scifact"),patch.object(harness_run,"GenericDataLoader",return_value=MagicMock(load=MagicMock(return_value=(corpus,q,qr))))]
            with patch.object(harness_run,"parse_arguments",return_value=Namespace(**base)), patch.object(harness_run,"DeployableKnowledgeClient",return_value=client):
                for x in common: x.start()
                try: harness_run.main()
                finally:
                    for x in common: x.stop()
            mapping["applicationToBeir"] = {"changed":"document"}; source.write_text(json.dumps(mapping))
            bad = {**base,"resume":True,"prepare_only":False}
            with patch.object(harness_run,"parse_arguments",return_value=Namespace(**bad)), patch.object(harness_run,"DeployableKnowledgeClient",side_effect=AssertionError("client constructed")):
                with self.assertRaises(RuntimeError): harness_run.main()

    def test_main_resumes_interrupted_search_without_reingestion(self) -> None:
        corpus = {
            "document": {
                "title": "Test document",
                "text": "one two three four five six",
            }
        }
        queries = {"q1": "query one", "q2": "query two"}
        qrels = {query_id: {"document": 1} for query_id in queries}
        first_client = FakeClient(interrupt_on="query two")
        resumed_client = FakeClient()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mapping.json"
            source.write_text(json.dumps({"beirToApplication":{"document":"app-document"},"applicationToBeir":{"app-document":"document"},"applicationToBeirAliases":{},"missingCorpusQrelDocumentIds":[],"skippedBeirDocuments":{}}))
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
                        Namespace(**{**vars(self.arguments(resume=False, run_name="resume-test")),"reuse_existing_runtime":True,"reuse_document_mapping":source,"runtime_fingerprint":"fixture"}),
                        Namespace(**{**vars(self.arguments(resume=True, run_name="resume-test")),"reuse_existing_runtime":True,"reuse_document_mapping":source,"runtime_fingerprint":"fixture"}),
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
                self.assertEqual(list(partial), ["q1"])
                self.assertFalse((run_directory / "metrics.json").exists())

                harness_run.main()

            self.assertEqual(first_client.ingest_calls, [])
            self.assertEqual(resumed_client.ingest_calls, [])
            self.assertEqual(
                first_client.search_calls,
                ["query one", "query two"],
            )
            self.assertEqual(resumed_client.search_calls, ["query two"])

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

    def test_legacy_main_resumes_interrupted_search_without_reingestion(self) -> None:
        corpus={"document":{"title":"Test document","text":"one two three four five six"}}
        queries={"q1":"query one","q2":"query two","q3":"query three"}; qrels={q:{"document":1} for q in queries}
        first=FakeClient(interrupt_on="query three"); resumed=FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); loader=MagicMock(); loader.load.return_value=(corpus,queries,qrels)
            with patch.object(harness_run,"RUNS_ROOT",root/"runs"), patch.object(harness_run,"DATASETS_ROOT",root/"datasets"), patch.object(harness_run,"download_dataset",return_value=root/"datasets/scifact"), patch.object(harness_run,"GenericDataLoader",return_value=loader), patch.object(harness_run,"DeployableKnowledgeClient",side_effect=[first,resumed]), patch.object(harness_run,"parse_arguments",side_effect=[self.arguments(resume=False,run_name="legacy"),self.arguments(resume=True,run_name="legacy")]), patch.object(harness_run,"evaluate_method",return_value={"validated":True}):
                with self.assertRaises(KeyboardInterrupt): harness_run.main()
                harness_run.main()
            self.assertEqual(len(first.ingest_calls),1); self.assertEqual(resumed.ingest_calls,[])
            self.assertEqual(json.loads((root/"runs/scifact-legacy/run-status.json").read_text())["status"],"complete")

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
            self.assertEqual(len(checkpoint_path.read_text().splitlines()), 1)

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

    def test_pdf_chunks_collapse_to_pages(self) -> None:
        collapsed = collapse_chunks_to_documents(
            [
                {"documentId": "app-1", "pageIndex": 11},
                {"documentId": "app-1", "pageIndex": 11},
                {"documentId": "app-1", "pageIndex": 3},
                {"documentId": "app-2", "pageIndex": 0},
            ],
            {"app-1": "manual", "app-2": "guide"},
            3,
            page_units=True,
        )
        self.assertEqual(list(collapsed), ["manual#p12", "manual#p4", "guide#p1"])

    def test_pdf_chunks_collapse_to_documents_on_a_document_split(self) -> None:
        collapsed = collapse_chunks_to_documents(
            [
                {"documentId": "app-1", "pageIndex": 11},
                {"documentId": "app-1", "pageIndex": 3},
                {"documentId": "app-2", "pageIndex": 0},
            ],
            {"app-1": "manual", "app-2": "guide"},
            3,
            page_units=False,
        )
        self.assertEqual(list(collapsed), ["manual", "guide"])

    def test_pdf_ingestion_sets_parsing_and_checks_page_counts(self) -> None:
        manifest = {"documents": [{"id": "manual", "file": "pdfs/manual.pdf", "pages": 3}]}
        client = MagicMock()
        client.set_table_recognition.return_value = {"pdfPageCleanup": False}
        client.ingest_file.return_value = {"documentId": "app-1", "pageCount": 3, "chunkCount": 7}
        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "mapping.json"
            harness_run.ingest_pdf_corpus(client, Path(directory), manifest, "docling", mapping_path)
            mapping = json.loads(mapping_path.read_text())
            client.set_table_recognition.assert_called_once_with(True)
            client.search.assert_not_called()
            self.assertEqual(mapping["unit"], "page")
            self.assertEqual(mapping["documentParsing"], "docling")
            self.assertEqual(mapping["applicationToBeir"], {"app-1": "manual"})
            harness_run.load_document_mapping(mapping_path, {"manual"})
            client.ingest_file.return_value = {"documentId": "app-1", "pageCount": 2, "chunkCount": 7}
            with self.assertRaises(RuntimeError):
                harness_run.ingest_pdf_corpus(client, Path(directory), manifest, "standard",
                                              mapping_path)
            client.set_table_recognition.assert_called_with(False)

    def test_pdf_cleanup_parsing_needs_a_server_with_cleanups(self) -> None:
        manifest = {"documents": [{"id": "manual", "file": "pdfs/manual.pdf", "pages": 3}]}
        client = MagicMock()
        client.set_table_recognition.return_value = {"pdfPageCleanup": False}
        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "mapping.json"
            with self.assertRaisesRegex(RuntimeError, "PDF cleanups"):
                harness_run.ingest_pdf_corpus(client, Path(directory), manifest,
                                              "docling-cleanup", mapping_path)
            client.ingest_file.assert_not_called()
            client.set_table_recognition.return_value = {}
            with self.assertRaisesRegex(RuntimeError, "PDF cleanups"):
                harness_run.ingest_pdf_corpus(client, Path(directory), manifest, "docling",
                                              mapping_path)
            client.set_table_recognition.return_value = {"pdfPageCleanup": True}
            client.ingest_file.return_value = {"documentId": "app-1", "pageCount": 3,
                                               "chunkCount": 7}
            harness_run.ingest_pdf_corpus(client, Path(directory), manifest, "docling-cleanup",
                                          mapping_path)
            client.set_table_recognition.assert_called_with(True)
            client.search.assert_not_called()
            self.assertEqual(json.loads(mapping_path.read_text())["documentParsing"],
                             "docling-cleanup")

    def test_pdf_prepare_recovers_committed_upload_before_mapping_was_written(self) -> None:
        import hashlib
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_pdf = root / "first.pdf"
            old_pdf.write_bytes(b"committed paper")
            digest = hashlib.sha256(old_pdf.read_bytes()).hexdigest()
            managed = f"documents/{digest[:16]}.pdf"
            (root / "documents").mkdir()
            (root / managed).write_bytes(old_pdf.read_bytes())
            (root / "second.pdf").write_bytes(b"pending paper")
            manifest = {"documents": [
                {"id": "a", "file": "first.pdf", "pages": 3},
                {"id": "b", "file": "second.pdf", "pages": 5},
            ]}
            with sqlite3.connect(root / "app.db") as db:
                db.execute("CREATE TABLE documents (id TEXT, source_path TEXT)")
                db.execute("INSERT INTO documents VALUES (?, ?)", ("app-a", managed))
            client = MagicMock()
            client.set_table_recognition.return_value = {"pdfPageCleanup": True}
            client.ingest_file.return_value = {"documentId": "app-b", "pageCount": 5, "chunkCount": 9}
            with patch.object(harness_run, "count_pages", return_value={old_pdf: 3}) as pages:
                mapping, _, _, _ = harness_run.ingest_pdf_corpus(
                    client, root, manifest, "docling-cleanup", root / "mapping.json", root)
            pages.assert_called_once_with([old_pdf])
            client.ingest_file.assert_called_once_with(root / "second.pdf")
            self.assertEqual(mapping, {"a": "app-a", "b": "app-b"})
            old_pdf.write_bytes(b"changed after interruption")
            with self.assertRaisesRegex(RuntimeError, "outside the current corpus"):
                harness_run.committed_pdfs(root, root, manifest["documents"])

    def test_only_cleanup_parsing_sets_the_server_flag(self) -> None:
        self.assertEqual(parsing_environment("docling-cleanup"), {"RAG_PDF_PAGE_CLEANUP": "true"})
        for mode in ("standard", "docling", None):
            self.assertEqual(parsing_environment(mode), {})

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
