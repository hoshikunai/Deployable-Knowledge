"""Unit tests for the bench CLI/GUI helpers. Run with:

    benchmarks/beir/.venv/bin/python -m unittest discover -s benchmarks/bench-gui
"""

from __future__ import annotations

import json
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
import scores
from bench import parse_arguments, parse_seeds
from run_suite import chunking_environment, pipeline_environment
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
            "reranker": "ettin-32m", "retrievalMultiplier": 5,
            "rerankMultiplier": 2, "rrfRankConstant": 60, "rerankMaxTokens": 512,
        })

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
        self.assertEqual(namespace.corpus, {"runtimeId": "runtime-scifact-001", "chunking": None})

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

    def test_prepare_needs_token_chunking(self) -> None:
        with self.assertRaises(ValueError):
            jobs.create_prepare_jobs(["nfcorpus"], jobs.Settings())

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


class PipelineLabelTests(unittest.TestCase):
    def test_describes_depths(self) -> None:
        config = make_spec().pipeline_config()
        self.assertEqual(pipeline_key(config), "rrf-ettin")
        self.assertEqual(describe_pipeline(config, 20), "RRF + Ettin 32M · 100→40→20")
        self.assertIn("Unknown", describe_pipeline(None, 20))
        wide = {**config, "rerankMaxTokens": 1024}
        self.assertIn("reranker reads 1024 tokens", describe_pipeline(wide, 20))


class CliTests(unittest.TestCase):
    def test_no_command_opens_gui(self) -> None:
        self.assertEqual(parse_arguments([]).command, "gui")

    def test_seed_parsing(self) -> None:
        self.assertEqual(parse_seeds("42, 314,2718"), [42, 314, 2718])


class ScoreTests(unittest.TestCase):
    def fake_run(self) -> SimpleNamespace:
        return SimpleNamespace(qrels={"q1": {"d1": 1, "d2": 0}},
                               application_to_beir={"a1": "d1", "a2": "d2", "self": "q1"})

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

    def test_markdown_table(self) -> None:
        self.assertEqual(scores.markdown_table(["A", "B"], [["1", "2"]]),
                         ["| A | B |", "|---|---|", "| 1 | 2 |"])


if __name__ == "__main__":
    unittest.main()
