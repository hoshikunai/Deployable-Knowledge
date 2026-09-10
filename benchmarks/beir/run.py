from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from app_client import DeployableKnowledgeClient


HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
RUNS_ROOT = HARNESS_ROOT / "runs"

METHODS = ("bm25", "semantic", "hybrid")
K_VALUES = [1, 3, 5, 10]


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
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Run only the first N queries; zero runs the complete split.",
    )
    return parser.parse_args()


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

    if arguments.search_depth < max(K_VALUES):
        raise ValueError(
            f"--search-depth must be at least {max(K_VALUES)}"
        )

    client = DeployableKnowledgeClient(arguments.base_url)
    client.heartbeat()

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{arguments.dataset}.zip"
    )

    dataset_path = util.download_and_unzip(
        dataset_url,
        str(DATASETS_ROOT),
    )

    corpus, queries, qrels = GenericDataLoader(
        data_folder=dataset_path
    ).load(split=arguments.split)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_directory = RUNS_ROOT / f"{arguments.dataset}-{run_id}"
    run_directory.mkdir(parents=True)

    beir_to_application: dict[str, str] = {}
    application_to_beir: dict[str, str] = {}

    print(f"Ingesting {len(corpus)} BEIR documents...")

    for index, (beir_document_id, document) in enumerate(
        corpus.items(),
        start=1,
    ):
        title = document.get("title", "").strip() or beir_document_id
        content = corpus_content(document)

        if not content:
            raise RuntimeError(
                f"BEIR document {beir_document_id} has no searchable content"
            )

        ingestion = client.ingest_document(
            title=title,
            text=content,
        )
        application_document_id = ingestion["documentId"]

        previous_beir_id = application_to_beir.get(
            application_document_id
        )
        if (
            previous_beir_id is not None
            and previous_beir_id != beir_document_id
        ):
            raise RuntimeError(
                "Two BEIR documents mapped to the same application "
                "document. They may contain identical content: "
                f"{previous_beir_id}, {beir_document_id}"
            )

        beir_to_application[beir_document_id] = (
            application_document_id
        )
        application_to_beir[application_document_id] = (
            beir_document_id
        )

        if index % 100 == 0 or index == len(corpus):
            print(f"Ingested {index}/{len(corpus)} documents")

    mapping_path = run_directory / "document-id-mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "beirToApplication": beir_to_application,
                "applicationToBeir": application_to_beir,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    selected_queries = list(queries.items())
    if arguments.query_limit > 0:
        selected_queries = selected_queries[
            : arguments.query_limit
        ]

    selected_qrels = {
        query_id: qrels[query_id]
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
                print(
                    f"Searched {index}/{len(selected_queries)} queries"
                )

    metrics = {
        method: evaluate_method(
            qrels=selected_qrels,
            results=results[method],
        )
        for method in METHODS
    }

    (run_directory / "document-rankings.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_directory / "run-config.json").write_text(
        json.dumps(
            {
                "dataset": arguments.dataset,
                "split": arguments.split,
                "baseUrl": arguments.base_url,
                "searchDepth": arguments.search_depth,
                "kValues": K_VALUES,
                "queryCount": len(selected_queries),
                "corpusCount": len(corpus),
                "titleProtocol": "title-plus-text",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {run_directory}")


if __name__ == "__main__":
    main()