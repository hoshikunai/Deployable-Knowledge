from __future__ import annotations

import argparse
import copy
import json
import random
import re
import os
import tempfile
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from app_client import DeployableKnowledgeClient
from dataset_catalog import download_dataset, validate_dataset_name


HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
RUNS_ROOT = HARNESS_ROOT / "runs"

METHODS = ("bm25", "semantic", "hybrid")
K_VALUES = [1, 3, 5, 10]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PUBLIC_MAPPING_SHA256 = "2f151186adc7433b2ce415c01d45965f59a5e6bbfe55f9478bf701b284d41905"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def validate_resume_configuration(
    saved: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"saved={saved.get(key)!r}, expected={value!r}"
            )


def record_query_checkpoint(
    path: Path,
    query_id: str,
    rankings: dict[str, dict[str, float]],
    elapsed_seconds: float,
    failure: str | None = None,
    *,
    raw_ranking: dict[str, Any] | None = None,
    attempt_count: int = 1,
    attempted_http_top_k: list[int] | None = None,
    final_http_top_k_by_method: dict[str, int] | None = None,
) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing = [record for record in existing if record["queryId"] != query_id]
    existing.append(
        {
            "queryId": query_id,
            "rankings": rankings,
            "rawRanking": raw_ranking,
            "elapsedSeconds": elapsed_seconds,
            "attemptCount": attempt_count,
            "failure": failure,
            "attemptedHttpTopK": attempted_http_top_k or [],
            "finalHttpTopKByMethod": final_http_top_k_by_method or {},
        }
    )
    atomic_write_jsonl(path, existing)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def load_query_checkpoints(
    path: Path,
    expected_query_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    allowed_query_ids = set(expected_query_ids)
    checkpoints: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read query checkpoints: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed query checkpoint on line {line_number}: {path}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Query checkpoint line {line_number} is not an object"
            )

        query_id = record.get("queryId")
        if not isinstance(query_id, str) or query_id not in allowed_query_ids:
            raise RuntimeError(
                f"Query checkpoint contains unknown query ID: {query_id!r}"
            )
        if query_id in checkpoints:
            raise RuntimeError(
                f"Query checkpoint contains duplicate query ID: {query_id}"
            )

        failure = record.get("failure")
        if failure is not None and not isinstance(failure, str):
            raise RuntimeError(
                f"Query checkpoint has invalid failure for {query_id}"
            )
        attempt_count = record.get("attemptCount")
        if not isinstance(attempt_count, int) or attempt_count < 1:
            raise RuntimeError(
                f"Query checkpoint has invalid attempt count for {query_id}"
            )

        if failure is None:
            rankings = record.get("rankings")
            raw_ranking = record.get("rawRanking")
            if not isinstance(rankings, dict) or set(rankings) != set(METHODS):
                raise RuntimeError(
                    f"Successful checkpoint lacks complete rankings for {query_id}"
                )
            if not isinstance(raw_ranking, dict):
                raise RuntimeError(
                    f"Successful checkpoint lacks raw ranking for {query_id}"
                )
            raw_methods = raw_ranking.get("methods")
            if not isinstance(raw_methods, dict) or set(raw_methods) != set(
                METHODS
            ):
                raise RuntimeError(
                    f"Successful checkpoint lacks raw methods for {query_id}"
                )
            for method in METHODS:
                ranking = rankings[method]
                if not isinstance(ranking, dict) or not all(
                    isinstance(document_id, str)
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    for document_id, score in ranking.items()
                ):
                    raise RuntimeError(
                        f"Checkpoint has invalid {method} ranking for {query_id}"
                    )

        checkpoints[query_id] = record

    return checkpoints


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Deployable Knowledge retrieval with BEIR."
    )
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:4179",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=100,
        help="Number of chunks requested before document-level collapsing.",
    )
    query_selection = parser.add_mutually_exclusive_group()
    query_selection.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help=(
            "Run only the first N dataset-ordered queries; zero runs the "
            "complete split."
        ),
    )
    query_selection.add_argument(
        "--query-counts",
        help=(
            "Comma-separated nested sample sizes, such as "
            "10,50,100,all. Queries are deterministically shuffled once, "
            "and each checkpoint reuses the preceding searches."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --query-counts (default: 42).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional stable artifact suffix supplied by a suite runner.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--search-timeout", type=float, default=180)
    parser.add_argument("--ingestion-timeout", type=float, default=1800)
    parser.add_argument("--chunk-overfetch-factor", type=int, default=2)
    parser.add_argument("--public-protocol", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--required-document-depth", type=int, default=10)
    parser.add_argument("--initial-http-top-k", type=int, default=20)
    parser.add_argument("--maximum-http-top-k", type=int, default=160)
    parser.add_argument("--reuse-document-mapping", type=Path)
    return parser.parse_args()


def validate_public_protocol(
    protocol: dict[str, Any], corpus: dict[str, Any], queries: dict[str, str],
    qrels: dict[str, dict[str, int]], selected_ids: list[str],
) -> None:
    if protocol.get("protocolVersion") != "scifact-full-test-v1":
        raise ValueError("Public protocol version mismatch")
    expected = {
        "datasetName": "scifact", "split": "test", "corpusDocumentCount": 5183,
        "queryCount": 300, "sampling": "none", "resultDepth": 10,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Public protocol requirement failed: {key}")
    systems = protocol.get("applicationSystems")
    if (not isinstance(systems, list) or len(systems) != 3
            or len(set(systems)) != 3 or set(systems) != set(METHODS)):
        raise ValueError("Public protocol application systems are invalid")
    if len(corpus) != 5183 or len(queries) != 300 or len(qrels) != 300:
        raise ValueError("Public protocol dataset counts are invalid")
    qrel_ids = list(qrels)
    if set(queries) != set(qrels) or selected_ids != qrel_ids:
        raise ValueError("Public protocol query IDs do not equal ordered test qrels")
    if protocol.get("metrics", {}).get("cutoffs") != K_VALUES:
        raise ValueError("Public protocol evaluation cutoffs are invalid")


def validate_public_mapping(
    mapping: dict[str, Any], corpus: dict[str, Any], qrels: dict[str, dict[str, int]]
) -> None:
    b2a = mapping.get("beirToApplication", {})
    a2b = mapping.get("applicationToBeir", {})
    if (len(b2a), len(a2b)) != (5183, 5183):
        raise RuntimeError("Public protocol requires 5,183 identity mappings")
    if mapping.get("applicationToBeirAliases") or mapping.get("skippedBeirDocuments"):
        raise RuntimeError("Public protocol forbids duplicate or skipped mappings")
    if set(b2a) != set(corpus) or set(a2b.values()) != set(corpus):
        raise RuntimeError("Public protocol mapping does not cover the corpus")
    if any(a2b.get(app_id) != beir_id for beir_id, app_id in b2a.items()):
        raise RuntimeError("Public protocol mapping is not one-to-one")
    if set().union(*[set(j) for j in qrels.values()]) - set(b2a):
        raise RuntimeError("Public protocol qrels contain unmapped documents")


def validate_public_results(
    results: dict[str, dict[str, dict[str, float]]], query_ids: list[str],
    corpus_ids: set[str], qrels: dict[str, dict[str, int]],
) -> None:
    if set(results) != set(METHODS):
        raise RuntimeError("Public protocol results lack an application method")
    for method in METHODS:
        if list(results[method]) != query_ids:
            raise RuntimeError("Public protocol method query ordering differs")
        for query_id in query_ids:
            ranking = results[method][query_id]
            if len(ranking) != 10 or len(set(ranking)) != 10:
                raise RuntimeError(f"{method} returned fewer than 10 unique documents for {query_id}")
            if set(ranking) - corpus_ids:
                raise RuntimeError(f"{method} returned an unknown document")
    if set(qrels) != set(query_ids):
        raise RuntimeError("Public protocol qrels do not cover the query manifest")


def query_manifest_fingerprint(query_ids: list[str], protocol: dict[str, Any]) -> str:
    payload = json.dumps(
        {"queryIds": query_ids, "protocol": protocol},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_name(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value}")
    return value


def parse_query_counts(raw_counts: str, available: int) -> list[int]:
    if available <= 0:
        raise ValueError("The selected split contains no queries")

    counts: list[int] = []
    for raw_value in raw_counts.split(","):
        value = raw_value.strip().lower()
        if not value:
            raise ValueError("--query-counts contains an empty value")

        if value == "all":
            count = available
        else:
            try:
                count = int(value)
            except ValueError as exc:
                raise ValueError(
                    "--query-counts values must be positive integers or 'all'"
                ) from exc

        if count <= 0:
            raise ValueError("--query-counts values must be positive")
        if count > available:
            raise ValueError(
                f"Requested {count} queries, but the split only has {available}"
            )
        counts.append(count)

    return sorted(set(counts))


def select_queries(
    queries: dict[str, str],
    query_limit: int,
    query_counts: str | None,
    sample_seed: int,
) -> tuple[list[tuple[str, str]], list[int], str]:
    available_queries = list(queries.items())

    if query_limit < 0:
        raise ValueError("--query-limit cannot be negative")

    if query_counts:
        checkpoint_counts = parse_query_counts(
            query_counts,
            len(available_queries),
        )
        if query_counts.strip().lower() == "all":
            return available_queries, checkpoint_counts, "dataset-order"
        selected_queries = sorted(
            available_queries,
            key=lambda item: item[0],
        )
        random.Random(sample_seed).shuffle(selected_queries)
        return (
            selected_queries[: checkpoint_counts[-1]],
            checkpoint_counts,
            "seeded-shuffle",
        )

    if query_limit > len(available_queries):
        raise ValueError(
            f"Requested {query_limit} queries, but the split only has "
            f"{len(available_queries)}"
        )

    selected_queries = (
        available_queries[:query_limit]
        if query_limit > 0
        else available_queries
    )
    return selected_queries, [len(selected_queries)], "dataset-order"


def corpus_content(document: dict[str, str]) -> str:
    """
    Use BEIR-like indexing semantics by making both title and text searchable.
    """
    title = document.get("title", "").strip()
    text = document.get("text", "").strip()
    return "\n\n".join(value for value in (title, text) if value)


def collapse_chunks_to_documents(
    hits: list[dict[str, Any]],
    application_to_beir: dict[str, str],
    target_count: int,
) -> dict[str, float]:
    """
    Preserve the rank of the first/highest chunk from each document.

    BEIR evaluates document IDs, while Deployable Knowledge returns chunk IDs.
    The artificial score is only used to preserve rank order.
    """
    ranked_documents: dict[str, float] = {}

    for chunk_rank, hit in enumerate(hits):
        application_document_id = hit["documentId"]

        if application_document_id not in application_to_beir:
            raise RuntimeError(
                "Search returned a document outside the isolated BEIR corpus: "
                f"{application_document_id}"
            )

        beir_document_id = application_to_beir[application_document_id]

        if beir_document_id in ranked_documents:
            continue

        ranked_documents[beir_document_id] = float(len(hits) - chunk_rank)

        if len(ranked_documents) == target_count:
            break

    return ranked_documents


def register_document_mapping(
    beir_document_id: str,
    application_document_id: str,
    beir_to_application: dict[str, str],
    application_to_beir: dict[str, str],
    application_to_beir_aliases: dict[str, list[str]],
) -> None:
    beir_to_application[beir_document_id] = application_document_id
    canonical_beir_id = application_to_beir.get(application_document_id)

    if canonical_beir_id is None:
        application_to_beir[application_document_id] = beir_document_id
        return

    application_to_beir_aliases.setdefault(
        application_document_id,
        [canonical_beir_id],
    ).append(beir_document_id)


def canonicalize_qrels(
    qrels: dict[str, dict[str, int]],
    beir_to_application: dict[str, str],
    application_to_beir: dict[str, str],
    missing_corpus_document_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Merge judgments for byte-identical BEIR documents.

    The application content-deduplicates documents, so two BEIR IDs can map
    to one application ID. Evaluating both IDs separately would penalize the
    application for being unable to return duplicate content twice.
    """
    canonical_qrels: dict[str, dict[str, int]] = {}
    allowed_missing_ids = missing_corpus_document_ids or set()

    for query_id, judgments in qrels.items():
        canonical_judgments: dict[str, int] = {}

        for beir_document_id, relevance in judgments.items():
            application_document_id = beir_to_application.get(
                beir_document_id
            )
            if application_document_id is None:
                if beir_document_id in allowed_missing_ids:
                    canonical_judgments[beir_document_id] = max(
                        relevance,
                        canonical_judgments.get(
                            beir_document_id,
                            relevance,
                        ),
                    )
                    continue
                raise RuntimeError(
                    "Qrels reference a document that was not ingested: "
                    f"{beir_document_id}"
                )

            canonical_beir_id = application_to_beir[
                application_document_id
            ]
            canonical_judgments[canonical_beir_id] = max(
                relevance,
                canonical_judgments.get(canonical_beir_id, relevance),
            )

        canonical_qrels[query_id] = canonical_judgments

    return canonical_qrels


def canonicalize_rankings(
    rankings: dict[str, dict[str, float]],
    beir_to_application: dict[str, str],
    application_to_beir: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Map external BEIR rankings to the application's canonical IDs."""
    transformed: dict[str, dict[str, float]] = {}
    for query_id, ranking in rankings.items():
        canonical: dict[str, float] = {}
        for document_id, score in ranking.items():
            application_id = beir_to_application.get(document_id)
            if application_id is None:
                raise RuntimeError(
                    "Ranking references a document outside the ingested corpus: "
                    f"{document_id}"
                )
            canonical_id = application_to_beir[application_id]
            if canonical_id not in canonical:
                canonical[canonical_id] = score
        transformed[query_id] = canonical
    return transformed


def evaluate_method(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    # BEIR may remove identical query/document IDs during evaluation, so give
    # each evaluator its own copy.
    ndcg, mean_average_precision, recall, precision = (
        EvaluateRetrieval.evaluate(
            qrels,
            copy.deepcopy(results),
            K_VALUES,
        )
    )

    mrr = EvaluateRetrieval.evaluate_custom(
        qrels,
        copy.deepcopy(results),
        K_VALUES,
        metric="mrr",
    )

    return {
        "ndcg": ndcg,
        "map": mean_average_precision,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def load_document_mapping(
    path: Path,
    corpus: dict[str, dict[str, str]],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    set[str],
    dict[str, dict[str, Any]],
]:
    mapping = load_json_object(path, "document ID mapping")
    beir_to_application = mapping.get("beirToApplication")
    application_to_beir = mapping.get("applicationToBeir")
    aliases = mapping.get("applicationToBeirAliases")
    missing_ids = mapping.get("missingCorpusQrelDocumentIds")
    skipped_documents = mapping.get("skippedBeirDocuments")

    if not isinstance(beir_to_application, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in beir_to_application.items()
    ):
        raise RuntimeError("Invalid beirToApplication document mapping")
    if not isinstance(application_to_beir, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in application_to_beir.items()
    ):
        raise RuntimeError("Invalid applicationToBeir document mapping")
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(alias, str) for alias in value)
        for key, value in aliases.items()
    ):
        raise RuntimeError("Invalid applicationToBeirAliases document mapping")
    if not isinstance(missing_ids, list) or not all(
        isinstance(value, str) for value in missing_ids
    ):
        raise RuntimeError("Invalid missingCorpusQrelDocumentIds mapping value")
    if not isinstance(skipped_documents, dict):
        raise RuntimeError("Invalid skippedBeirDocuments mapping value")

    mapped_or_skipped = set(beir_to_application) | set(skipped_documents)
    if mapped_or_skipped != set(corpus):
        raise RuntimeError(
            "Document mapping does not describe the complete selected corpus"
        )
    for beir_id, application_id in beir_to_application.items():
        canonical_id = application_to_beir.get(application_id)
        if canonical_id not in beir_to_application:
            raise RuntimeError(
                f"Document mapping has no canonical BEIR ID for {beir_id}"
            )

    return (
        beir_to_application,
        application_to_beir,
        aliases,
        set(missing_ids),
        skipped_documents,
    )


def ingest_corpus(
    client: DeployableKnowledgeClient,
    corpus: dict[str, dict[str, str]],
    judged_document_ids: set[str],
    missing_corpus_document_ids: set[str],
    mapping_path: Path,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
]:
    beir_to_application: dict[str, str] = {}
    application_to_beir: dict[str, str] = {}
    application_to_beir_aliases: dict[str, list[str]] = {}
    skipped_documents: dict[str, dict[str, Any]] = {}

    print(f"Ingesting {len(corpus)} BEIR documents...")
    for index, (beir_document_id, document) in enumerate(
        corpus.items(), start=1
    ):
        title = document.get("title", "").strip() or beir_document_id
        content = corpus_content(document)
        word_count = len(content.split())

        if word_count < 5:
            if beir_document_id in judged_document_ids:
                raise RuntimeError(
                    "A judged BEIR document cannot be indexed because it "
                    f"contains only {word_count} word(s): {beir_document_id}"
                )
            skipped_documents[beir_document_id] = {
                "reason": "fewer-than-five-words",
                "wordCount": word_count,
            }
            print(
                f"Skipped unjudged BEIR document {beir_document_id}: "
                f"only {word_count} word(s)"
            )
            continue

        ingestion = client.ingest_document(title=title, text=content)
        register_document_mapping(
            beir_document_id=beir_document_id,
            application_document_id=ingestion["documentId"],
            beir_to_application=beir_to_application,
            application_to_beir=application_to_beir,
            application_to_beir_aliases=application_to_beir_aliases,
        )
        if index % 100 == 0 or index == len(corpus):
            print(f"Ingested {index}/{len(corpus)} documents")

    atomic_write_json(
        mapping_path,
        {
            "beirToApplication": beir_to_application,
            "applicationToBeir": application_to_beir,
            "applicationToBeirAliases": application_to_beir_aliases,
            "missingCorpusQrelDocumentIds": sorted(
                missing_corpus_document_ids
            ),
            "skippedBeirDocuments": skipped_documents,
        },
    )
    return (
        beir_to_application,
        application_to_beir,
        application_to_beir_aliases,
        skipped_documents,
    )


def build_query_result(
    query_id: str,
    query_text: str,
    response: dict[str, Any],
    application_to_beir: dict[str, str],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    rankings: dict[str, dict[str, float]] = {}
    raw_record: dict[str, Any] = {
        "queryId": query_id,
        "query": query_text,
        "methods": {},
    }
    for method in METHODS:
        hits = response.get(method)
        if not isinstance(hits, list):
            raise RuntimeError(f"Search response lacks {method} results")
        raw_record["methods"][method] = [
            {
                "chunkId": hit["chunkId"],
                "documentId": hit["documentId"],
                "chunkIndex": hit["chunkIndex"],
            }
            for hit in hits
        ]
        rankings[method] = collapse_chunks_to_documents(
            hits=hits,
            application_to_beir=application_to_beir,
            target_count=max(K_VALUES),
        )
    return rankings, raw_record


def adaptive_document_depth_search(
    client: DeployableKnowledgeClient,
    query_id: str,
    query_text: str,
    application_to_beir: dict[str, str],
    initial_http_top_k: int = 20,
    maximum_http_top_k: int = 160,
    required_document_depth: int = 10,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if initial_http_top_k < 1 or maximum_http_top_k < initial_http_top_k:
        raise ValueError("Invalid adaptive HTTP depth bounds")
    depth = initial_http_top_k
    attempts: list[int] = []
    best: dict[str, dict[str, float]] = {}
    raw: dict[str, Any] | None = None
    final_depths: dict[str, int] = {}
    started = time.monotonic()
    while True:
        attempts.append(depth)
        response = client.search(query=query_text, top_k=depth)
        rankings, raw_record = build_query_result(
            query_id, query_text, response, application_to_beir
        )
        raw = raw_record
        for method in METHODS:
            if method not in final_depths and len(rankings[method]) >= required_document_depth:
                best[method] = dict(list(rankings[method].items())[:required_document_depth])
                final_depths[method] = depth
        if len(final_depths) == len(METHODS):
            return best, {"rawRanking": raw, "attemptedHttpTopK": attempts,
                          "finalHttpTopKByMethod": final_depths,
                          "elapsedSeconds": time.monotonic() - started}
        if depth >= maximum_http_top_k:
            deficient = [method for method in METHODS if method not in final_depths]
            raise RuntimeError(
                f"Adaptive retrieval exhausted at HTTP topK {depth}; "
                f"methods below {required_document_depth}: {deficient}"
            )
        next_depth = min(depth * 2, maximum_http_top_k)
        if next_depth == depth:
            raise RuntimeError("Adaptive retrieval cannot increase HTTP depth")
        depth = next_depth


def run_query_searches(
    client: DeployableKnowledgeClient,
    selected_queries: list[tuple[str, str]],
    application_to_beir: dict[str, str],
    http_top_k: int,
    checkpoint_path: Path,
    status_path: Path,
    require_complete: bool = False,
    initial_http_top_k: int = 20,
    maximum_http_top_k: int = 160,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selected_query_ids = [query_id for query_id, _ in selected_queries]
    checkpoints = load_query_checkpoints(
        checkpoint_path, selected_query_ids
    )
    completed_query_ids = {
        query_id
        for query_id, record in checkpoints.items()
        if record["failure"] is None
    }

    print(f"Searching {len(selected_queries)} queries...")
    for index, (query_id, query_text) in enumerate(selected_queries, start=1):
        if query_id in completed_query_ids:
            continue

        previous_attempts = checkpoints.get(query_id, {}).get(
            "attemptCount", 0
        )
        started_at = time.monotonic()
        rankings: dict[str, dict[str, float]] = {
            method: {} for method in METHODS
        }
        raw_record: dict[str, Any] | None = None
        attempted_depths: list[int] = []
        final_depths: dict[str, int] = {}
        errors: list[str] = []
        attempts_this_invocation = 0

        for _attempt in range(2):
            attempts_this_invocation += 1
            try:
                if require_complete:
                    rankings, adaptive_record = adaptive_document_depth_search(
                        client, query_id, query_text, application_to_beir,
                        initial_http_top_k, maximum_http_top_k, max(K_VALUES),
                    )
                    raw_record = adaptive_record["rawRanking"]
                    attempted_depths = adaptive_record["attemptedHttpTopK"]
                    final_depths = adaptive_record["finalHttpTopKByMethod"]
                else:
                    response = client.search(query=query_text, top_k=http_top_k)
                    rankings, raw_record = build_query_result(
                        query_id, query_text, response, application_to_beir
                    )
                if require_complete and any(
                    len(rankings[method]) != max(K_VALUES) for method in METHODS
                ):
                    raise RuntimeError("public protocol requires exactly 10 unique documents")
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        elapsed_seconds = time.monotonic() - started_at
        failure = None if raw_record is not None else " | ".join(errors)
        record_query_checkpoint(
            checkpoint_path,
            query_id,
            rankings,
            elapsed_seconds,
            failure,
            raw_ranking=raw_record,
            attempt_count=previous_attempts + attempts_this_invocation,
            attempted_http_top_k=attempted_depths,
            final_http_top_k_by_method=final_depths,
        )
        checkpoints = load_query_checkpoints(
            checkpoint_path, selected_query_ids
        )
        if failure is None:
            completed_query_ids.add(query_id)

        failed_query_ids = [
            saved_query_id
            for saved_query_id in selected_query_ids
            if saved_query_id in checkpoints
            and checkpoints[saved_query_id]["failure"] is not None
        ]
        atomic_write_json(
            status_path,
            {
                "status": "searching",
                "completedQueryIds": [
                    saved_query_id
                    for saved_query_id in selected_query_ids
                    if saved_query_id in completed_query_ids
                ],
                "failedQueryIds": failed_query_ids,
            },
        )
        if index % 10 == 0 or index == len(selected_queries):
            print(f"Processed {index}/{len(selected_queries)} queries")

    checkpoints = load_query_checkpoints(checkpoint_path, selected_query_ids)
    failures: list[dict[str, Any]] = []
    for query_id in selected_query_ids:
        checkpoint = checkpoints.get(query_id)
        if checkpoint is None:
            failures.append(
                {
                    "queryId": query_id,
                    "attemptCount": 0,
                    "error": "query has no checkpoint",
                }
            )
        elif checkpoint["failure"] is not None:
            failures.append(
                {
                    "queryId": query_id,
                    "attemptCount": checkpoint["attemptCount"],
                    "error": checkpoint["failure"],
                }
            )
    if failures:
        return ({method: {} for method in METHODS}, [], failures)

    results = {
        method: {
            query_id: checkpoints[query_id]["rankings"][method]
            for query_id in selected_query_ids
        }
        for method in METHODS
    }
    raw_rankings = [
        checkpoints[query_id]["rawRanking"] for query_id in selected_query_ids
    ]
    return results, raw_rankings, []


def main() -> None:
    arguments = parse_arguments()

    dataset_name = validate_dataset_name(arguments.dataset)
    if arguments.split not in {"train", "dev", "test"}:
        raise ValueError("--split must be train, dev, or test")
    if arguments.run_name:
        validate_name(arguments.run_name, RUN_NAME_PATTERN, "run name")
    if arguments.resume and not arguments.run_name:
        raise ValueError("--resume requires --run-name")
    if arguments.resume and getattr(arguments, "reuse_document_mapping", None):
        raise ValueError("--reuse-document-mapping cannot be combined with --resume")
    if arguments.chunk_overfetch_factor < 1:
        raise ValueError("--chunk-overfetch-factor must be at least 1")

    if arguments.search_depth < max(K_VALUES):
        raise ValueError(
            f"--search-depth must be at least {max(K_VALUES)}"
        )

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_path = download_dataset(dataset_name, DATASETS_ROOT)
    corpus, queries, qrels = GenericDataLoader(
        data_folder=dataset_path
    ).load(split=arguments.split)

    public_protocol: dict[str, Any] | None = None
    protocol_path = getattr(arguments, "public_protocol", None)
    if protocol_path:
        public_protocol = load_json_object(protocol_path, "public protocol")

    run_id = arguments.run_name or datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    run_directory = RUNS_ROOT / f"{dataset_name}-{run_id}"
    if arguments.resume:
        if not run_directory.is_dir():
            raise RuntimeError(
                f"Cannot resume missing run directory: {run_directory}"
            )
    else:
        if run_directory.exists() and any(run_directory.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite existing run: {run_directory}"
            )
        run_directory.mkdir(parents=True, exist_ok=True)

    selected_queries, checkpoint_counts, selection_protocol = select_queries(
        queries,
        arguments.query_limit,
        arguments.query_counts,
        arguments.sample_seed,
    )
    if public_protocol is not None:
        public_query_ids = list(qrels)
        selected_queries = [(query_id, queries[query_id]) for query_id in public_query_ids]
        checkpoint_counts = [len(selected_queries)]
        selection_protocol = "public-qrels-order"
    selected_query_ids = [query_id for query_id, _ in selected_queries]
    if public_protocol is not None:
        validate_public_protocol(public_protocol, corpus, queries, qrels, selected_query_ids)
    http_top_k = (
        arguments.search_depth * arguments.chunk_overfetch_factor
    )
    immutable_protocol = {
        "schema": "beir-run-v2",
        "dataset": dataset_name,
        "split": arguments.split,
        "queryIds": selected_query_ids,
        "queryCounts": checkpoint_counts,
        "selectionProtocol": selection_protocol,
        "sampleSeed": (
            arguments.sample_seed
            if selection_protocol == "seeded-shuffle"
            else None
        ),
        "searchDepth": arguments.search_depth,
        "httpTopK": http_top_k,
        "chunkOverfetchFactor": arguments.chunk_overfetch_factor,
        "methods": list(METHODS),
        "kValues": K_VALUES,
        "titleProtocol": "title-plus-text",
        "duplicateCanonicalization": "application-id-mapping-v1",
        "corpusCount": len(corpus),
        "persistentRuntimeRequiredForResume": True,
    }
    immutable_protocol["queryManifestFingerprint"] = query_manifest_fingerprint(
        selected_query_ids, immutable_protocol
    )

    config_path = run_directory / "run-config.json"
    selection_path = run_directory / "query-selection.json"
    status_path = run_directory / "run-status.json"
    mapping_path = run_directory / "document-id-mapping.json"
    checkpoint_path = run_directory / "query-checkpoints.jsonl"

    if arguments.resume:
        if not config_path.is_file():
            raise RuntimeError("Cannot resume without run-config.json")
        if not selection_path.is_file():
            raise RuntimeError("Cannot resume without query-selection.json")
        if not mapping_path.is_file():
            raise RuntimeError("Cannot resume without document-id-mapping.json")
        saved_protocol = load_json_object(config_path, "run configuration")
        validate_resume_configuration(saved_protocol, immutable_protocol)
        saved_selection = load_json_object(
            selection_path, "query selection"
        )
        if saved_selection.get("queryIds") != selected_query_ids:
            raise ValueError("Resume query manifest does not match run config")
    else:
        atomic_write_json(config_path, immutable_protocol)
        atomic_write_json(
            selection_path,
            {
                "queryIds": selected_query_ids,
                "checkpointCounts": checkpoint_counts,
                "protocol": selection_protocol,
                "sampleSeed": immutable_protocol["sampleSeed"],
            },
        )

    if public_protocol is not None:
        protocol_hash = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
        atomic_write_json(
            run_directory / "execution-config.json",
            {
                "runName": run_id, "resultDepth": 10,
                "chunkOverfetchFactor": arguments.chunk_overfetch_factor,
                "httpTopK": http_top_k, "searchTimeout": arguments.search_timeout,
                "ingestionTimeout": arguments.ingestion_timeout,
                "dataset": dataset_name, "split": arguments.split,
                "querySelection": selection_protocol, "methods": list(METHODS),
                "protocolSha256": protocol_hash,
            },
        )
        atomic_write_json(
            status_path,
            {
                "status": "starting",
                "completedQueryIds": [],
                "failedQueryIds": [],
            },
        )

    judged_document_ids = {
        document_id
        for judgments in qrels.values()
        for document_id in judgments
    }
    missing_corpus_document_ids = judged_document_ids - set(corpus)
    if missing_corpus_document_ids:
        print(
            "Warning: qrels reference "
            f"{len(missing_corpus_document_ids)} document(s) absent from "
            "the corpus; those relevance judgments remain in the "
            "evaluation and cannot be retrieved."
        )

    client = DeployableKnowledgeClient(
        arguments.base_url,
        search_timeout=arguments.search_timeout,
        ingestion_timeout=arguments.ingestion_timeout,
    )
    client.heartbeat()

    if arguments.resume:
        (
            beir_to_application,
            application_to_beir,
            application_to_beir_aliases,
            saved_missing_ids,
            skipped_documents,
        ) = load_document_mapping(mapping_path, corpus)
        if saved_missing_ids != missing_corpus_document_ids:
            raise RuntimeError(
                "Saved mapping does not match missing corpus qrel IDs"
            )
    elif getattr(arguments, "reuse_document_mapping", None):
        mapping_source = arguments.reuse_document_mapping
        if (not mapping_source.is_file()
                or hashlib.sha256(mapping_source.read_bytes()).hexdigest() != PUBLIC_MAPPING_SHA256):
            raise RuntimeError("Mapping reuse file or validated hash is invalid")
        (
            beir_to_application, application_to_beir,
            application_to_beir_aliases, saved_missing_ids, skipped_documents,
        ) = load_document_mapping(mapping_source, corpus)
        mapping_path = mapping_source
        if (len(beir_to_application) != 5183 or application_to_beir_aliases
                or skipped_documents or saved_missing_ids):
            raise RuntimeError("Mapping reuse is not a complete identity mapping")
        atomic_write_json(config_path, {**immutable_protocol,
            "mappingSource": str(mapping_source),
            "mappingSha256": PUBLIC_MAPPING_SHA256})
    else:
        (
            beir_to_application,
            application_to_beir,
            application_to_beir_aliases,
            skipped_documents,
        ) = ingest_corpus(
            client,
            corpus,
            judged_document_ids,
            missing_corpus_document_ids,
            mapping_path,
        )

    if public_protocol is not None:
        mapping_value = load_json_object(mapping_path, "document ID mapping")
        validate_public_mapping(mapping_value, corpus, qrels)
    if getattr(arguments, "prepare_only", False):
        atomic_write_json(status_path, {"status": "prepared", "completedQueryIds": [], "failedQueryIds": []})
        print(f"Prepared benchmark runtime at {run_directory}")
        return

    canonical_qrels = canonicalize_qrels(
        qrels=qrels,
        beir_to_application=beir_to_application,
        application_to_beir=application_to_beir,
        missing_corpus_document_ids=missing_corpus_document_ids,
    )
    if application_to_beir_aliases:
        duplicate_count = sum(
            len(aliases) - 1
            for aliases in application_to_beir_aliases.values()
        )
        print(
            f"Canonicalized {duplicate_count} duplicate BEIR document(s) "
            f"across {len(application_to_beir_aliases)} group(s)"
        )

    selected_qrels = {
        query_id: canonical_qrels[query_id]
        for query_id, _query in selected_queries
    }
    results, raw_rankings, failures = run_query_searches(
        client,
        selected_queries,
        application_to_beir,
        http_top_k,
        checkpoint_path,
        status_path,
        require_complete=public_protocol is not None,
        initial_http_top_k=getattr(arguments, "initial_http_top_k", http_top_k),
        maximum_http_top_k=getattr(arguments, "maximum_http_top_k", http_top_k),
    )
    failures_path = run_directory / "failures.json"
    atomic_write_json(failures_path, failures)
    if failures:
        atomic_write_json(
            status_path,
            {
                "status": "incomplete",
                "completedQueryIds": [
                    query_id
                    for query_id in selected_query_ids
                    if query_id
                    not in {failure["queryId"] for failure in failures}
                ],
                "failedQueryIds": [
                    failure["queryId"] for failure in failures
                ],
            },
        )
        raise RuntimeError(
            f"Benchmark incomplete: {len(failures)} query failure(s); "
            "resume with the same arguments and --resume"
        )

    if public_protocol is not None:
        validate_public_results(results, selected_query_ids, set(corpus), qrels)

    metrics_by_query_count: dict[str, dict[str, Any]] = {}
    for query_count in checkpoint_counts:
        checkpoint_query_ids = selected_query_ids[:query_count]
        checkpoint_qrels = {
            query_id: selected_qrels[query_id]
            for query_id in checkpoint_query_ids
        }
        checkpoint_results = {
            method: {
                query_id: results[method][query_id]
                for query_id in checkpoint_query_ids
            }
            for method in METHODS
        }
        metrics_by_query_count[str(query_count)] = {
            method: evaluate_method(
                qrels=checkpoint_qrels,
                results=checkpoint_results[method],
            )
            for method in METHODS
        }

    metrics = metrics_by_query_count[str(checkpoint_counts[-1])]

    checkpoints = load_query_checkpoints(
        checkpoint_path, selected_query_ids
    )
    atomic_write_jsonl(
        run_directory / "chunk-rankings.jsonl", raw_rankings
    )
    atomic_write_json(run_directory / "document-rankings.json", results)
    atomic_write_json(run_directory / "metrics.json", metrics)
    atomic_write_json(
        run_directory / "metrics-by-query-count.json",
        metrics_by_query_count,
    )
    atomic_write_json(
        run_directory / "query-timing.json",
        {
            "totalSeconds": sum(
                checkpoints[query_id]["elapsedSeconds"]
                for query_id in selected_query_ids
            ),
            "queries": [
                {
                    "queryId": query_id,
                    "elapsedSeconds": checkpoints[query_id][
                        "elapsedSeconds"
                    ],
                    "attemptCount": checkpoints[query_id][
                        "attemptCount"
                    ],
                }
                for query_id in selected_query_ids
            ],
        },
    )
    atomic_write_json(
        run_directory / "run-summary.json",
        {
            "queryCount": len(selected_queries),
            "indexedDocumentCount": len(application_to_beir),
            "duplicateDocumentCount": (
                len(corpus)
                - len(application_to_beir)
                - len(skipped_documents)
            ),
            "missingCorpusQrelDocumentCount": len(
                missing_corpus_document_ids
            ),
            "skippedDocumentCount": len(skipped_documents),
            "baseUrlForFinalInvocation": arguments.base_url,
        },
    )
    atomic_write_json(
        status_path,
        {
            "status": "complete",
            "completedQueryIds": selected_query_ids,
            "failedQueryIds": [],
        },
    )

    print(json.dumps(metrics, indent=2))
    if len(checkpoint_counts) > 1:
        print(f"Query checkpoints evaluated: {checkpoint_counts}")
    print(f"Artifacts written to {run_directory}")


if __name__ == "__main__":
    main()
