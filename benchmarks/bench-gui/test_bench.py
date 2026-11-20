"""Unit tests for the bench CLI/GUI helpers. Run with:

    benchmarks/beir/.venv/bin/python -m unittest discover -s benchmarks/bench-gui
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import paths  # noqa: F401  (puts the BEIR harness on sys.path)
import corpora
import jobs
import run_suite
import scores
from bench import parse_arguments, parse_seeds
from benchmarks.benchmark_lease import PREPARE_MIN_AVAILABLE, QUERY_MIN_AVAILABLE
from run_suite import (chunking_environment, harness_options, link_directory, pipeline_environment,
                       vulkan_environment)
from runs import describe_pipeline, pipeline_key

FIXED_TIME = datetime(2026, 9, 28, 12, 0, 0, 123456, tzinfo=timezone.utc)


def make_spec(pipeline: str = "rrf-ettin", queries: str = "100", **settings: int) -> jobs.JobSpec:
    return jobs.JobSpec(dataset="scifact", pipeline=pipeline, queries=queries, seed=42,
                        settings=jobs.Settings(**settings))


class RunNameTests(unittest.TestCase):
    def test_reranked_sample_run_name_encodes_every_depth(self) -> None:
        name = jobs.make_run_name(make_spec(), FIXED_TIME)
        self.assertEqual(name, "rrf-ettin-p100-s40-o20-100q-seed42-20260928-120000-123456")

    def test_rrf_full_run_omits_shortlist_and_seed(self) -> None:
        name = jobs.make_run_name(make_spec("rrf", "all"), FIXED_TIME)
        self.assertEqual(name, "rrf-p100-o20-all-20260928-120000-123456")

    def test_dedup_modes_are_named(self) -> None:
        self.assertIn("-dx-", jobs.make_run_name(make_spec(dedupMode="exact"), FIXED_TIME))
        self.assertIn("-dn-", jobs.make_run_name(make_spec(dedupMode="near"), FIXED_TIME))
        name = jobs.make_run_name(make_spec(dedupMode="document-cap", maxChunksPerDocument=3),
                                  FIXED_TIME)
        self.assertIn("-dc3-", name)
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"dedupMode": "fuzzy"})

    def test_non_default_rrf_constant_is_recorded(self) -> None:
        name = jobs.make_run_name(make_spec("rrf", "all", rrfRankConstant=20), FIXED_TIME)
        self.assertIn("-k20-", name)


class SettingsTests(unittest.TestCase):
    def test_chunk_settings_validation(self) -> None:
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"chunkOverlapTokens": 10})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"chunkMaxTokens": 128, "chunkOverlapTokens": 128})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"chunkMaxTokens": 8})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"embeddingModel": "unknown-model"})
        self.assertIsNone(jobs.Settings().chunking())
        self.assertEqual(jobs.Settings(chunkMaxTokens=256, chunkOverlapTokens=0).chunking(),
                         {"maxTokens": 256, "overlapTokens": 0})

    def test_rejects_unknown_and_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"surprise": 1})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"documentDepth": 5})

    def test_partial_settings_keep_defaults(self) -> None:
        settings = jobs.Settings.from_dict({"rerankMultiplier": 3})
        self.assertEqual(settings.rerankMultiplier, 3)
        self.assertEqual(settings.retrievalMultiplier, 5)


class JobTests(unittest.TestCase):
    def test_job_round_trips_through_json(self) -> None:
        spec = make_spec()
        spec.runName = "example"
        restored = jobs.JobSpec.from_dict(spec.to_dict())
        self.assertEqual(restored, spec)
        self.assertEqual(restored.run_id, "scifact-example")

    def test_pipeline_config_matches_app_contract(self) -> None:
        self.assertEqual(make_spec().pipeline_config(), {
            "fusion": "rrf", "reranker": "ettin-32m", "retrievalMultiplier": 5,
            "rerankMultiplier": 2, "rrfRankConstant": 60, "rerankMaxTokens": 512,
            "dedup": "none", "maxChunksPerDocument": 2,
            "rrfSemanticWeight": 1.0, "rrfBm25Weight": 1.0, "fusionLevel": "chunk",
            "documentOrder": "none", "rerankTitle": True,
        })

    def test_default_pipeline_pins_app_defaults(self) -> None:
        spec = make_spec("default", dedupMode="near", retrievalMultiplier=5)
        config = spec.pipeline_config()
        self.assertEqual((config["fusion"], config["reranker"]), ("union", "ettin-32m"))
        self.assertEqual(config["retrievalMultiplier"], 2)
        self.assertEqual(config["rerankMaxTokens"], 512)
        self.assertFalse(config["rerankTitle"])
        self.assertEqual(config["dedup"], "none")
        name = jobs.make_run_name(spec, datetime(2026, 9, 28, 12, 0, 0, 123456, timezone.utc))
        self.assertEqual(name, "default-p40-o20-100q-seed42-20260928-120000-123456")
        self.assertEqual(pipeline_environment(config)["RAG_HYBRID_FUSION"], "union")

    def test_fusion_settings_are_named_and_validated(self) -> None:
        spec = make_spec("rrf-ettin", "all")
        spec.settings = jobs.Settings(rrfSemanticWeight=1.0, rrfBm25Weight=0.5,
                                      fusionLevel="document", documentOrder="sum-top2",
                                      rerankTitle="off", split="train")
        name = jobs.make_run_name(spec, FIXED_TIME)
        for tag in ("-w1x0.5-", "-fdoc-", "-os2-", "-nt-", "-train-"):
            self.assertIn(tag, name)
        self.assertFalse(spec.pipeline_config()["rerankTitle"])
        self.assertTrue(jobs.Settings().rerankTitle == "on")
        self.assertNotIn("-nt-", jobs.make_run_name(
            jobs.JobSpec("scifact", "rrf", "all", 42, jobs.Settings(rerankTitle="off")), FIXED_TIME))
        for bad in ({"rrfSemanticWeight": 0, "rrfBm25Weight": 0}, {"rrfBm25Weight": -1},
                    {"fusionLevel": "sentence"}, {"split": "validation"}):
            with self.assertRaises(ValueError):
                jobs.Settings.from_dict(bad)

    def test_split_must_exist_for_the_dataset(self) -> None:
        with patch.object(corpora, "require_prepared"):
            jobs.validate_job("scifact", "rrf", "all", jobs.Settings(split="train"))
            with self.assertRaises(ValueError):
                jobs.validate_job("arguana", "rrf", "all", jobs.Settings(split="train"))

    def test_split_reaches_the_harness(self) -> None:
        spec = make_spec("rrf")
        spec.settings = jobs.Settings(split="dev")
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            self.assertEqual(jobs.suite_arguments(spec, resume=False).split, "dev")

    def test_full_runs_ignore_extra_seeds(self) -> None:
        with patch.object(jobs, "validate_job"):
            created = jobs.create_jobs(["scifact"], "rrf", "all", [42, 314], jobs.Settings())
            sampled = jobs.create_jobs(["scifact"], "rrf", "100", [42, 314], jobs.Settings())
        self.assertEqual(len(created), 1)
        self.assertEqual([spec.seed for spec in sampled], [42, 314])

    def test_suite_arguments_carry_pipeline_and_memory_ceiling(self) -> None:
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            namespace = jobs.suite_arguments(make_spec(), resume=True)
        self.assertTrue(namespace.resume)
        self.assertEqual(namespace.query_counts, "100")
        self.assertEqual(namespace.pipeline["reranker"], "ettin-32m")
        self.assertEqual(namespace.memory_ceiling, 7 * jobs.GIGABYTE)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-001")
        self.assertEqual(namespace.corpus, {"runtimeId": "runtime-scifact-001", "chunking": None,
                                            "embeddingModel": "nomic-embed-text-v1.5"})
        self.assertEqual(namespace.embedding_model, "nomic-embed-text-v1.5")

    def test_embedding_model_selects_its_own_corpus_copy(self) -> None:
        spec = make_spec("rrf")
        spec.settings = jobs.Settings(embeddingModel="bge-small-en-v1.5")
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-bge-small")
        self.assertEqual(namespace.embedding_model, "bge-small-en-v1.5")
        self.assertIn("-bge-small-", jobs.make_run_name(spec, FIXED_TIME))
        both = jobs.Settings(chunkMaxTokens=256, chunkOverlapTokens=32,
                             embeddingModel="bge-small-en-v1.5")
        self.assertEqual(corpora.runtime_id("scifact", both.chunking(), both.embeddingModel),
                         "runtime-scifact-t256o32-bge-small")

    def test_document_parsing_selects_its_own_pdf_corpus_copy(self) -> None:
        spec = jobs.JobSpec(dataset="pdf-eval", pipeline="default", queries="all", seed=42,
                            settings=jobs.Settings(documentParsing="docling"))
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(namespace.runtime_id, "runtime-pdf-eval-docling")
        self.assertEqual(namespace.document_parsing, "docling")
        self.assertEqual(namespace.corpus["documentParsing"], "docling")
        self.assertIn("-docling-", jobs.make_run_name(spec, FIXED_TIME))
        self.assertNotIn("documentParsing", make_spec("rrf").corpus())
        self.assertEqual(corpora.runtime_id("pdf-eval", None), "runtime-pdf-eval-001")
        cleanup = jobs.JobSpec(dataset="pdf-eval", pipeline="default", queries="all", seed=42,
                               settings=jobs.Settings(documentParsing="docling-cleanup"))
        self.assertEqual(cleanup.runtime_id, "runtime-pdf-eval-docling-cleanup")
        self.assertIn("-docling-cleanup-", jobs.make_run_name(cleanup, FIXED_TIME))
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"documentParsing": "layout"})

    def test_embedding_device_selects_its_own_corpus_copy(self) -> None:
        spec = make_spec("rrf")
        spec.settings = jobs.Settings(embeddingDevice="vulkan", vulkanDevice=1)
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-vulkan")
        self.assertEqual(namespace.embedding_device, "vulkan")
        self.assertEqual(namespace.vulkan_device, 1)
        self.assertEqual(namespace.corpus["embeddingDevice"], "vulkan")
        self.assertIn("-vulkan-", jobs.make_run_name(spec, FIXED_TIME))
        self.assertNotIn("embeddingDevice", make_spec("rrf").corpus())
        self.assertEqual(make_spec("rrf").runtime_id, "runtime-scifact-001")
        self.assertEqual(
            corpora.runtime_id("pdf-eval", {"maxTokens": 256, "overlapTokens": 32},
                               parsing="docling", device="cpu"),
            "runtime-pdf-eval-t256o32-cpu-docling")

    def test_embedding_device_settings_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"embeddingDevice": "metal"})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"embeddingDevice": "cpu",
                                     "embeddingModel": "bge-small-en-v1.5"})
        for index in (-2, 8):
            with self.assertRaises(ValueError):
                jobs.Settings.from_dict({"vulkanDevice": index})
        self.assertEqual(jobs.Settings.from_dict({"vulkanDevice": "0"}).vulkanDevice, 0)

    def test_prepare_job_carries_the_embedding_device(self) -> None:
        settings = jobs.Settings(embeddingDevice="cpu")
        with patch.object(corpora, "variant_status", return_value={"state": "missing"}):
            [spec] = jobs.create_prepare_jobs(["scifact"], settings)
        namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-cpu")
        self.assertEqual(namespace.embedding_device, "cpu")
        self.assertEqual(namespace.vulkan_device, -1)

    def test_harness_receives_the_vulkan_device_the_server_sees(self) -> None:
        spec = make_spec("rrf")
        spec.settings = jobs.Settings(embeddingDevice="vulkan", vulkanDevice=1)
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            options = harness_options(jobs.suite_arguments(spec, resume=False), None)
        self.assertEqual(options[options.index("--vulkan-device") + 1], "1")
        self.assertEqual(options[options.index("--embedding-device") + 1], "vulkan")
        spec.settings = jobs.Settings(embeddingDevice="vulkan")
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            options = harness_options(jobs.suite_arguments(spec, resume=False), None)
        self.assertNotIn("--vulkan-device", options)

    def test_query_jobs_start_with_less_free_memory_than_prepares(self) -> None:
        for prepare_only, minimum in ((True, PREPARE_MIN_AVAILABLE), (False, QUERY_MIN_AVAILABLE)):
            with patch.object(run_suite, "memory_preflight",
                              side_effect=RuntimeError("preflight")) as preflight, \
                    self.assertRaisesRegex(RuntimeError, "preflight"):
                run_suite.run_dataset("scifact", SimpleNamespace(prepare_only=prepare_only), "suite")
            preflight.assert_called_once_with(minimum)
        self.assertLess(QUERY_MIN_AVAILABLE, PREPARE_MIN_AVAILABLE)

    def test_docling_parsing_is_unavailable_for_text_corpora(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch.object(corpora.paths, "DATASETS_ROOT", Path(root)):
            dataset = Path(root) / "scifact"
            (dataset / "qrels").mkdir(parents=True)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
                (dataset / name).write_text("")
            for parsing in ("docling", "docling-cleanup"):
                status = corpora.variant_status("scifact", None, parsing=parsing)
                self.assertEqual(status["state"], "unavailable")
            with self.assertRaises(ValueError):
                corpora.start_preparing("scifact", None, parsing="docling")

    def test_embedding_models_have_their_own_copies_and_limits(self) -> None:
        tags = {model: corpora.variant_id(None, model) for model in corpora.EMBEDDING_MODELS}
        self.assertEqual(tags, {"nomic-embed-text-v1.5": "001", "bge-small-en-v1.5": "bge-small"})
        jobs.Settings(embeddingModel="bge-small-en-v1.5", chunkMaxTokens=512)
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"embeddingModel": "bge-small-en-v1.5", "chunkMaxTokens": 1024})
        with self.assertRaises(ValueError):
            jobs.Settings.from_dict({"chunkMaxTokens": 4096})

    def test_prepare_allows_a_new_embedding_model_with_default_chunks(self) -> None:
        settings = jobs.Settings(embeddingModel="bge-small-en-v1.5")
        with patch.object(corpora, "variant_status", return_value={"state": "missing"}):
            [spec] = jobs.create_prepare_jobs(["scifact"], settings)
        namespace = jobs.suite_arguments(spec, resume=False)
        self.assertIsNone(namespace.chunking)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-bge-small")

    def test_token_chunked_jobs_use_their_corpus_copy(self) -> None:
        spec = make_spec(chunkMaxTokens=256, chunkOverlapTokens=32)
        with patch.object(corpora, "require_prepared", return_value=Path("/tmp/mapping.json")):
            namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(namespace.runtime_id, "runtime-scifact-t256o32")
        self.assertIn("-t256o32-", jobs.make_run_name(spec, FIXED_TIME))

    def test_prepare_jobs_ingest_into_a_new_runtime(self) -> None:
        settings = jobs.Settings(chunkMaxTokens=256, chunkOverlapTokens=32)
        with patch.object(corpora, "variant_status", return_value={"state": "missing"}):
            [spec] = jobs.create_prepare_jobs(["nfcorpus"], settings)
        namespace = jobs.suite_arguments(spec, resume=False)
        self.assertEqual(spec.kind, "prepare")
        self.assertTrue(namespace.prepare_only)
        self.assertFalse(namespace.reuse_existing_runtime)
        self.assertEqual(namespace.chunking, {"maxTokens": 256, "overlapTokens": 32})
        self.assertEqual(namespace.runtime_id, "runtime-nfcorpus-t256o32")

    def test_prepare_skips_prepared_copies(self) -> None:
        with patch.object(corpora, "variant_status", return_value={"state": "prepared"}):
            with self.assertRaises(ValueError):
                jobs.create_prepare_jobs(["nfcorpus"], jobs.Settings())

    def test_existing_default_copy_is_never_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(root)), \
                patch.object(corpora, "dataset_files_present", return_value=True):
            runtime = corpora.start_preparing("fiqa", None)  # no default copy yet: allowed
            self.assertEqual(runtime.name, "runtime-fiqa-001")
            corpora.start_preparing("fiqa", None)  # an unfinished attempt can be retried
            corpora.write_marker(runtime, {"dataset": "fiqa", "state": "prepared"})
            with self.assertRaises(ValueError):
                corpora.start_preparing("fiqa", None)

    def test_rerank_token_limit_follows_the_reranker(self) -> None:
        settings = jobs.Settings(rerankMaxTokens=1024)
        with patch.object(corpora, "require_prepared"):
            jobs.validate_job("scifact", "rrf-ettin", "all", settings)
            with self.assertRaises(ValueError):
                jobs.validate_job("scifact", "rrf-msmarco", "all", settings)
        spec = make_spec(rerankMaxTokens=1024)
        self.assertIn("-r1024-", jobs.make_run_name(spec, FIXED_TIME))

    def test_server_environment_from_pipeline(self) -> None:
        environment = pipeline_environment(make_spec("rrf").pipeline_config())
        self.assertEqual(environment["RAG_HYBRID_RERANKER"], "none")
        self.assertEqual(environment["RAG_RRF_K"], "60")
        self.assertEqual(pipeline_environment(None), {})


class CorpusVariantTests(unittest.TestCase):
    def test_variant_names(self) -> None:
        self.assertEqual(corpora.runtime_id("arguana", None), "runtime-arguana-001")
        self.assertEqual(corpora.runtime_id("arguana", {"maxTokens": 128, "overlapTokens": 0}),
                         "runtime-arguana-t128o0")
        self.assertEqual(chunking_environment({"maxTokens": 128, "overlapTokens": 16}),
                         {"RAG_CHUNK_MAX_TOKENS": "128", "RAG_CHUNK_OVERLAP_TOKENS": "16"})
        self.assertEqual(chunking_environment(None), {})
        self.assertEqual(vulkan_environment(2), {"GGML_VK_VISIBLE_DEVICES": "2"})
        self.assertEqual(vulkan_environment(-1), {})
        self.assertEqual(vulkan_environment(None), {})

    def test_copy_embedded_on_another_device_is_not_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(root)), \
                patch.object(corpora, "dataset_files_present", return_value=True):
            runtime = corpora.start_preparing("scifact", None, device="cpu")
            self.assertEqual(runtime.name, "runtime-scifact-cpu")
            marker = json.loads((runtime / "beir-dataset.json").read_text())
            self.assertEqual(marker["embeddingDevice"], "cpu")
            corpora.write_marker(runtime, {**marker, "embeddingDevice": "vulkan",
                                           "state": "prepared"})
            status = corpora.variant_status("scifact", None, device="cpu")
            self.assertEqual(status["state"], "missing")
            self.assertIn("embedded on vulkan", status["reason"])
            self.assertEqual(corpora.copy_variant(marker)[-1], "cpu")
            self.assertEqual(corpora.copy_variant({"dataset": "scifact"})[-1], "auto")

    def test_start_preparing_only_replaces_unfinished_copies(self) -> None:
        chunking = {"maxTokens": 256, "overlapTokens": 32}
        with tempfile.TemporaryDirectory() as root, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(root)), \
                patch.object(corpora, "dataset_files_present", return_value=True):
            runtime = corpora.start_preparing("scifact", chunking)
            (runtime / "partial.db").write_text("x")
            corpora.start_preparing("scifact", chunking)  # an unfinished copy is rebuilt
            self.assertFalse((runtime / "partial.db").exists())
            corpora.write_marker(runtime, {"dataset": "scifact", "state": "prepared"})
            with self.assertRaises(ValueError):
                corpora.start_preparing("scifact", chunking)
            foreign = Path(root) / "runtime-nfcorpus-t256o32"
            foreign.mkdir()
            with self.assertRaises(ValueError):
                corpora.start_preparing("nfcorpus", chunking)
            self.assertEqual(json.loads((runtime / "beir-dataset.json").read_text())["state"],
                             "prepared")

    def test_retry_keeps_partial_pdf_database(self) -> None:
        with tempfile.TemporaryDirectory() as root, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(root)), \
                patch.object(corpora, "dataset_files_present", return_value=True), \
                patch.object(corpora, "is_pdf", return_value=True):
            runtime = corpora.start_preparing("papers", None, parsing="docling-cleanup")
            (runtime / "app.db").write_bytes(b"partial database")
            corpora.start_preparing("papers", None, parsing="docling-cleanup")
            self.assertEqual((runtime / "app.db").read_bytes(), b"partial database")


class PipelineLabelTests(unittest.TestCase):
    def test_describes_depths(self) -> None:
        config = make_spec().pipeline_config()
        self.assertEqual(pipeline_key(config), "rrf-ettin")
        self.assertEqual(describe_pipeline(config, 20), "RRF + Ettin 32M · 100→40→20")
        self.assertIn("Unknown", describe_pipeline(None, 20))
        wide = {**config, "rerankMaxTokens": 1024}
        self.assertIn("reranker reads 1024 tokens", describe_pipeline(wide, 20))
        tuned = {**config, "rrfBm25Weight": 0.5, "fusionLevel": "document",
                 "documentOrder": "sum-top2", "rerankTitle": False}
        label = describe_pipeline(tuned, 20)
        for text in ("RRF weights semantic 1 : BM25 0.5", "document-level RRF",
                     "ordered by document", "no title in reranker input"):
            self.assertIn(text, label)


class EnvironmentTests(unittest.TestCase):
    def test_environment_labels(self) -> None:
        from runs import describe_environment
        windows = {"platform": "windows", "serverEnvironment": {
            "embeddingBackend": "vulkan", "gpus": ["AMD Radeon RX 7700S"]}}
        self.assertEqual(describe_environment(windows), "Windows · Vulkan · AMD Radeon RX 7700S")
        wsl = {"platform": "wsl", "serverEnvironment": {"embeddingBackend": "vulkan", "gpus": []}}
        self.assertEqual(describe_environment(wsl), "WSL · CPU (no Vulkan GPU)")
        cpu = {"platform": "linux", "serverEnvironment": {"embeddingBackend": "cpu", "gpus": []}}
        self.assertEqual(describe_environment(cpu), "Linux · CPU")
        self.assertEqual(describe_environment({"platform": "wsl", "serverEnvironment": None}), "WSL")
        self.assertIsNone(describe_environment({"buildTime": 1}))
        self.assertIsNone(describe_environment(None))

    def test_machine_platform(self) -> None:
        with patch.object(jobs.sys, "platform", "win32"):
            self.assertEqual(jobs.machine_platform(), "windows")
        with patch.object(jobs.sys, "platform", "linux"), \
                patch.object(jobs.platform, "release", return_value="6.18.58.1-microsoft-standard-WSL2"):
            self.assertEqual(jobs.machine_platform(), "wsl")
        with patch.object(jobs.sys, "platform", "linux"), \
                patch.object(jobs.platform, "release", return_value="6.8.0-generic"):
            self.assertEqual(jobs.machine_platform(), "linux")

    def test_execution_meta_records_platform_and_server_environment(self) -> None:
        spec = make_spec("rrf")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "server-environment.json").write_text(json.dumps(
                {"embeddingBackend": "vulkan", "gpus": ["AMD Radeon RX 7700S"]}))
            with patch.object(jobs, "run_directory", return_value=run), \
                    patch.object(jobs, "build_status", return_value={"stale": False}), \
                    patch.object(jobs, "machine_platform", return_value="windows"):
                jobs.record_execution_meta(spec)
            meta = json.loads((run / "execution-meta.json").read_text())
        self.assertEqual(meta["platform"], "windows")
        self.assertEqual(meta["serverEnvironment"]["gpus"], ["AMD Radeon RX 7700S"])


class LauncherTests(unittest.TestCase):
    LAUNCHER = Path(__file__).resolve().parent / "launch.mjs"

    def test_launcher_runs_bench_and_passes_the_exit_code(self) -> None:
        import subprocess
        shown = subprocess.run(["node", str(self.LAUNCHER), "run", "--help"],
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("--pipeline", shown.stdout)
        failed = subprocess.run(["node", str(self.LAUNCHER), "no-such-command"],
                                capture_output=True, text=True, timeout=120)
        self.assertNotEqual(failed.returncode, 0)

    def test_launcher_explains_a_missing_venv(self) -> None:
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as root:
            copy = Path(root) / "benchmarks" / "bench-gui"
            copy.mkdir(parents=True)
            shutil.copy(self.LAUNCHER, copy / "launch.mjs")
            result = subprocess.run(["node", str(copy / "launch.mjs")], capture_output=True,
                                    text=True, timeout=60)
        self.assertEqual(result.returncode, 1)
        self.assertIn("venv is missing", result.stderr)


class CliTests(unittest.TestCase):
    def test_run_one_turns_ctrl_break_into_keyboard_interrupt(self) -> None:
        import signal
        import bench

        installed = {}
        with patch.object(signal, "SIGBREAK", 21, create=True), \
                patch.object(signal, "signal", lambda signum, handler: installed.update({signum: handler})), \
                patch.object(bench, "load_job"), patch.object(bench, "run_job", return_value="complete"):
            self.assertEqual(bench.command_run_one(SimpleNamespace(run_id="job")), 0)
        with self.assertRaises(KeyboardInterrupt):
            installed[21](21, None)

    def test_no_command_opens_gui(self) -> None:
        self.assertEqual(parse_arguments([]).command, "gui")

    def test_seed_parsing(self) -> None:
        self.assertEqual(parse_seeds("42, 314,2718"), [42, 314, 2718])


class ScoreTests(unittest.TestCase):
    def fake_run(self) -> SimpleNamespace:
        return SimpleNamespace(qrels={"q1": {"d1": 1, "d2": 0}},
                               application_to_beir={"a1": "d1", "a2": "d2", "self": "q1"},
                               page_units=False)

    def test_pdf_runs_label_chunks_by_page(self) -> None:
        run = SimpleNamespace(qrels={"q1": {"manual#p12": 1}}, application_to_beir={"a1": "manual"},
                              page_units=True)
        record = {"queryId": "q1", "methods": {"hybrid": [
            {"documentId": "a1", "pageIndex": 11, "score": 0.9},
            {"documentId": "a1", "pageIndex": 12, "score": 0.5},
        ]}}
        self.assertEqual(scores.labeled_chunks(run, record, "hybrid"), [(0.9, True), (0.5, False)])

    def test_labels_chunks_and_skips_the_query_document(self) -> None:
        record = {"queryId": "q1", "methods": {"hybrid": [
            {"documentId": "self", "score": 0.99},
            {"documentId": "a1", "score": 0.8},
            {"documentId": "a2", "score": 0.3},
        ]}}
        self.assertEqual(scores.labeled_chunks(self.fake_run(), record, "hybrid"),
                         [(0.8, True), (0.3, False)])

    def test_bm25_scores_are_relative_to_the_top_score(self) -> None:
        record = {"queryId": "q1", "methods": {"bm25": [
            {"documentId": "a1", "score": 8.0}, {"documentId": "a2", "score": 2.0},
        ]}}
        self.assertEqual(scores.labeled_chunks(self.fake_run(), record, "bm25"),
                         [(1.0, True), (0.25, False)])

    def test_missing_scores_mean_no_calibration(self) -> None:
        record = {"queryId": "q1", "methods": {"hybrid": [{"documentId": "a1", "score": None}]}}
        self.assertIsNone(scores.labeled_chunks(self.fake_run(), record, "hybrid"))

    def test_sign_flip_p_value(self) -> None:
        self.assertEqual(scores.sign_flip_p_value(np.zeros(10)), 1.0)
        self.assertLess(scores.sign_flip_p_value(np.full(40, 0.2)), 0.001)

    def test_recomputed_mrr_drops_the_query_document(self) -> None:
        run = SimpleNamespace(
            config={"kValues": [1, 10]}, query_ids=["q1", "q2"],
            qrels={"q1": {"d2": 1}, "q2": {"d9": 1}},
            rankings={"hybrid": {"q1": {"q1": 3.0, "d2": 2.0}, "q2": {"d5": 1.0}}},
        )
        # q1: own document skipped, answer at rank 1 -> 1.0; q2: no answer -> 0
        self.assertEqual(scores.recompute_mrr(run, "hybrid"), {"MRR@1": 0.5, "MRR@10": 0.5})

    def test_retriever_check_explains_expected_differences(self) -> None:
        self.assertEqual(scores.retriever_check(True, True, []), ("yes", False))
        self.assertEqual(scores.retriever_check(True, False, ["embedding model"]),
                         ("semantic differs by design (embedding model)", False))
        self.assertFalse(scores.retriever_check(False, False, ["chunk size"])[1])
        self.assertTrue(scores.retriever_check(False, True, ["embedding model"])[1])
        self.assertTrue(scores.retriever_check(True, False, [])[1])
        self.assertFalse(scores.retriever_check(False, False, ["document parsing"])[1])

    def test_markdown_table(self) -> None:
        self.assertEqual(scores.markdown_table(["A", "B"], [["1", "2"]]),
                         ["| A | B |", "|---|---|", "| 1 | 2 |"])


if __name__ == "__main__":
    unittest.main()


class CorpusCopyTests(unittest.TestCase):
    def make_copy(self, root: Path, name: str, marker: dict | None) -> Path:
        runtime = root / name
        runtime.mkdir()
        (runtime / "app.db").write_bytes(b"x" * 2048)
        if marker is not None:
            corpora.write_marker(runtime, marker)
        return runtime

    def test_copies_list_variants_runs_and_leftovers(self) -> None:
        import runs
        with tempfile.TemporaryDirectory() as runtime_root, \
                tempfile.TemporaryDirectory() as runs_root, \
                tempfile.TemporaryDirectory() as jobs_root, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(runtime_root)), \
                patch.object(runs.paths, "RUNS_ROOT", Path(runs_root)), \
                patch.object(jobs.paths, "JOBS_ROOT", Path(jobs_root)):
            root = Path(runtime_root)
            prepared = {"dataset": "pdf-eval", "chunking": None, "state": "prepared",
                        "embeddingModel": "nomic-embed-text-v1.5", "documentCount": 8,
                        "chunkCount": 3245, "documentParsing": "docling"}
            self.make_copy(root, "runtime-pdf-eval-docling", prepared)
            self.make_copy(root, "runtime-pdf-eval-docling-step1-cleanup", prepared)
            self.make_copy(root, "runtime-scifact-001", {"dataset": "scifact", "state": "prepared"})
            self.make_copy(root, "runtime-unrelated", None)
            for run_id, corpus, state in (
                    ("pdf-eval-default-docling-1", {"runtimeId": "runtime-pdf-eval-docling"}, "complete"),
                    ("pdf-eval-default-docling-2", {"runtimeId": "runtime-pdf-eval-docling"}, "failed"),
                    ("scifact-suite-old", None, None)):
                directory = Path(runs_root) / run_id
                directory.mkdir()
                config = {"dataset": run_id.split("-default")[0].split("-suite")[0]}
                if corpus:
                    config["corpus"] = corpus
                (directory / "run-config.json").write_text(json.dumps(config))
                if state:
                    (Path(jobs_root) / f"{run_id}.json").write_text(json.dumps({"state": state}))
            copies = {copy["runtimeId"]: copy for copy in runs.list_copies_with_runs()}
        self.assertEqual(set(copies), {"runtime-pdf-eval-docling",
                                       "runtime-pdf-eval-docling-step1-cleanup",
                                       "runtime-scifact-001"})
        docling = copies["runtime-pdf-eval-docling"]
        self.assertTrue(docling["regenerable"])
        self.assertFalse(docling["default"])
        self.assertEqual(docling["runCount"], 2)
        self.assertEqual(docling["resumableRuns"], ["pdf-eval-default-docling-2"])
        self.assertEqual(docling["documentParsing"], "docling")
        self.assertGreaterEqual(docling["sizeBytes"], 2048)
        self.assertFalse(copies["runtime-pdf-eval-docling-step1-cleanup"]["regenerable"])
        self.assertTrue(copies["runtime-scifact-001"]["default"])
        self.assertTrue(copies["runtime-scifact-001"]["regenerable"])
        self.assertEqual(copies["runtime-scifact-001"]["runCount"], 1)

    def test_delete_copy_refuses_non_copies_and_keeps_link_targets(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_root, \
                tempfile.TemporaryDirectory() as shared, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(runtime_root)):
            root = Path(runtime_root)
            runtime = self.make_copy(root, "runtime-scifact-t256o32",
                                     {"dataset": "scifact", "state": "prepared"})
            (Path(shared) / "model.onnx").write_text("weights")
            link_directory(runtime / "tables", Path(shared))
            self.make_copy(root, "runtime-plain", None)
            for name in ("..", "runtime-../x", "scifact", "runtime-plain", "runtime-missing"):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    corpora.delete_copy(name)
            corpora.delete_copy("runtime-scifact-t256o32")
            self.assertFalse(runtime.exists())
            self.assertTrue((Path(shared) / "model.onnx").is_file())
            self.assertTrue((root / "runtime-plain").is_dir())

    def test_junctions_to_shared_caches_are_never_followed_or_emptied(self) -> None:
        # On Windows the link is a real junction. Elsewhere a symlink stands in for one:
        # is_junction says yes, is_symlink says no, and rmdir removes the link.
        from contextlib import ExitStack
        real_rmdir = Path.rmdir
        removed: list[str] = []

        def fake_rmdir(path: Path) -> None:
            if path.name == "tables":
                removed.append(path.name)
                path.unlink()
                return
            real_rmdir(path)

        with tempfile.TemporaryDirectory() as runtime_root, \
                tempfile.TemporaryDirectory() as shared, \
                patch.object(corpora.paths, "RUNTIME_ROOT", Path(runtime_root)):
            runtime = self.make_copy(Path(runtime_root), "runtime-scifact-cpu",
                                     {"dataset": "scifact", "state": "prepared"})
            (Path(shared) / "model.onnx").write_text("x" * 5000)
            (runtime / ".cache").mkdir()
            link_directory(runtime / ".cache" / "tables", Path(shared))
            with ExitStack() as patches:
                if sys.platform != "win32":
                    patches.enter_context(patch.object(Path, "is_junction",
                                                       lambda path: path.name == "tables"))
                    patches.enter_context(patch.object(Path, "is_symlink", lambda path: False))
                    patches.enter_context(patch.object(Path, "rmdir", fake_rmdir))
                size = corpora.folder_size(runtime)
                corpora.delete_copy("runtime-scifact-cpu")
            self.assertLess(size, 5000)
            if sys.platform != "win32":
                self.assertEqual(removed, ["tables"])
            self.assertFalse(runtime.exists())
            self.assertEqual((Path(shared) / "model.onnx").read_text(), "x" * 5000)

    def test_regenerate_uses_the_copy_variant_not_saved_settings(self) -> None:
        marker = {"dataset": "scifact", "state": "prepared", "embeddingModel": "bge-small-en-v1.5",
                  "chunking": {"maxTokens": 256, "overlapTokens": 32}}
        settings = jobs.copy_settings(marker, jobs.Settings(chunkMaxTokens=0,
                                                            documentDepth=50))
        self.assertEqual(settings.chunking(), {"maxTokens": 256, "overlapTokens": 32})
        self.assertEqual(settings.embeddingModel, "bge-small-en-v1.5")
        self.assertEqual(settings.documentParsing, "standard")
        self.assertEqual(settings.documentDepth, 50)
        with patch.object(corpora, "variant_status", return_value={"state": "missing"}):
            spec = jobs.create_regenerate_job(marker, settings)
        self.assertEqual(spec.kind, "prepare")
        self.assertEqual(spec.runtime_id, "runtime-scifact-t256o32-bge-small")

    def test_copy_actions_refuse_while_a_job_uses_the_copy(self) -> None:
        import server
        from fastapi import HTTPException
        busy = SimpleNamespace(uses_runtime=lambda runtime: runtime == "runtime-scifact-001")
        with patch.object(server, "get_runner", return_value=busy), \
                patch.object(server, "delete_copy") as delete:
            for action in (server.remove_copy, server.regenerate_copy):
                with self.assertRaises(HTTPException) as raised:
                    action("runtime-scifact-001")
                self.assertEqual(raised.exception.status_code, 409)
            delete.assert_not_called()
