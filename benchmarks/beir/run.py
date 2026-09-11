from __future__ import annotations

import argparse
import copy
import json
import random
import re
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
    return parser.parse_args()


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


def main() -> None:
    arguments = parse_arguments()

    dataset_name = validate_dataset_name(arguments.dataset)
    if arguments.split not in {"train", "dev", "test"}:
        raise ValueError("--split must be train, dev, or test")
    if arguments.run_name:
        validate_name(arguments.run_name, RUN_NAME_PATTERN, "run name")

    if arguments.search_depth < max(K_VALUES):
        raise ValueError(
            f"--search-depth must be at least {max(K_VALUES)}"
        )

    client = DeployableKnowledgeClient(arguments.base_url)
    client.heartbeat()

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_path = download_dataset(dataset_name, DATASETS_ROOT)

    corpus, queries, qrels = GenericDataLoader(
        data_folder=dataset_path
    ).load(split=arguments.split)

    run_id = arguments.run_name or datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    run_directory = RUNS_ROOT / f"{dataset_name}-{run_id}"
    run_directory.mkdir(parents=True)

    beir_to_application: dict[str, str] = {}
    application_to_beir: dict[str, str] = {}
    application_to_beir_aliases: dict[str, list[str]] = {}

    print(f"Ingesting {len(corpus)} BEIR documents...")

    skipped_documents: dict[str, dict[str, Any]] = {}
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

    for index, (beir_document_id, document) in enumerate(
        corpus.items(),
        start=1,
    ):
        title = document.get("title", "").strip() or beir_document_id
        content = corpus_content(document)

        word_count = len(content.split())

        if word_count < 5:
            if beir_document_id in judged_document_ids:
                raise RuntimeError(
                    "A judged BEIR document cannot be indexed because it "
                    f"contains only {word_count} word(s): "
                    f"{beir_document_id}"
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

        ingestion = client.ingest_document(
            title=title,
            text=content,
        )
        application_document_id = ingestion["documentId"]

        register_document_mapping(
            beir_document_id=beir_document_id,
            application_document_id=application_document_id,
            beir_to_application=beir_to_application,
            application_to_beir=application_to_beir,
            application_to_beir_aliases=application_to_beir_aliases,
        )

        if index % 100 == 0 or index == len(corpus):
            print(f"Ingested {index}/{len(corpus)} documents")

    mapping_path = run_directory / "document-id-mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "beirToApplication": beir_to_application,
                "applicationToBeir": application_to_beir,
                "applicationToBeirAliases": (
                    application_to_beir_aliases
                ),
                "missingCorpusQrelDocumentIds": sorted(
                    missing_corpus_document_ids
                ),
                "skippedBeirDocuments": skipped_documents,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

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

    selected_queries, checkpoint_counts, selection_protocol = (
        select_queries(
            queries=queries,
            query_limit=arguments.query_limit,
            query_counts=arguments.query_counts,
            sample_seed=arguments.sample_seed,
        )
    )

    selected_qrels = {
        query_id: canonical_qrels[query_id]
        for query_id, _query in selected_queries
    }

    results: dict[str, dict[str, dict[str, float]]] = {
        method: {} for method in METHODS
    }

    raw_rankings_path = run_directory / "chunk-rankings.jsonl"

    print(f"Searching {len(selected_queries)} queries...")

    with raw_rankings_path.open("w", encoding="utf-8") as raw_file:
        for index, (query_id, query_text) in enumerate(
            selected_queries,
            start=1,
        ):
            response = client.search(
                query=query_text,
                top_k=arguments.search_depth,
            )

            raw_record: dict[str, Any] = {
                "queryId": query_id,
                "query": query_text,
                "methods": {},
            }

            for method in METHODS:
                hits = response.get(method, [])

                raw_record["methods"][method] = [
                    {
                        "chunkId": hit["chunkId"],
                        "documentId": hit["documentId"],
                        "chunkIndex": hit["chunkIndex"],
                    }
                    for hit in hits
                ]

                results[method][query_id] = (
                    collapse_chunks_to_documents(
                        hits=hits,
                        application_to_beir=application_to_beir,
                        target_count=max(K_VALUES),
                    )
                )

            raw_file.write(json.dumps(raw_record) + "\n")

            if index % 10 == 0 or index == len(selected_queries):
                raw_file.flush()
                print(
                    f"Searched {index}/{len(selected_queries)} queries"
                )

    selected_query_ids = [query_id for query_id, _ in selected_queries]
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

    (run_directory / "document-rankings.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_directory / "metrics-by-query-count.json").write_text(
        json.dumps(
            metrics_by_query_count,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_directory / "query-selection.json").write_text(
        json.dumps(
            {
                "checkpointCounts": checkpoint_counts,
                "protocol": selection_protocol,
                "sampleSeed": (
                    arguments.sample_seed
                    if selection_protocol == "seeded-shuffle"
                    else None
                ),
                "queryIds": selected_query_ids,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_directory / "run-config.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "split": arguments.split,
                "baseUrl": arguments.base_url,
                "searchDepth": arguments.search_depth,
                "kValues": K_VALUES,
                "queryCount": len(selected_queries),
                "queryCounts": checkpoint_counts,
                "querySelectionProtocol": selection_protocol,
                "sampleSeed": (
                    arguments.sample_seed
                    if selection_protocol == "seeded-shuffle"
                    else None
                ),
                "corpusCount": len(corpus),
                "indexedDocumentCount": len(application_to_beir),
                "duplicateDocumentCount": (
                    len(corpus)
                    - len(application_to_beir)
                    - len(skipped_documents)
                ),
                "missingCorpusQrelDocumentCount": len(
                    missing_corpus_document_ids
                ),
                "missingCorpusQrelDocumentIds": sorted(
                    missing_corpus_document_ids
                ),
                "skippedDocumentCount": len(skipped_documents),
                "skippedDocumentIds": sorted(skipped_documents),
                "titleProtocol": "title-plus-text",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    if len(checkpoint_counts) > 1:
        print(f"Query checkpoints evaluated: {checkpoint_counts}")
    print(f"Artifacts written to {run_directory}")


if __name__ == "__main__":
    main()
