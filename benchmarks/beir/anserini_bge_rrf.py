"""Evaluate an Anserini BM25 + BGE dense RRF baseline on a BEIR split."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, util
from run import evaluate_method

K_VALUES = [1, 3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-counts", default="all")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    return parser.parse_args()


def select_queries(queries: dict[str, str], raw: str, seed: int):
    import random

    ordered = sorted(queries.items())
    random.Random(seed).shuffle(ordered)
    counts = sorted({len(ordered) if x == "all" else int(x) for x in raw.split(",")})
    if counts[-1] > len(ordered):
        raise ValueError("query checkpoint exceeds split size")
    return ordered[: counts[-1]], counts


def rrf(*rankings: dict[str, list[str]], k: int = 60) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for ranking in rankings:
        for query_id, documents in ranking.items():
            for rank, document_id in enumerate(documents, start=1):
                scores.setdefault(query_id, {})
                scores[query_id][document_id] = scores[query_id].get(document_id, 0.0) + 1 / (k + rank)
    return {query_id: dict(sorted(values.items(), key=lambda item: -item[1])) for query_id, values in scores.items()}


def main() -> None:
    args = parse_args()
    dataset_path = args.data_root / args.dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=args.split)
    if args.query_manifest:
        manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
        query_ids = manifest["queryIds"]
        selected = [(query_id, queries[query_id]) for query_id in query_ids]
        counts = [len(selected)]
    else:
        selected, counts = select_queries(queries, args.query_counts, args.sample_seed)
    with tempfile.TemporaryDirectory(prefix="anserini-") as temporary:
        collection = Path(temporary) / "collection.jsonl"
        with collection.open("w", encoding="utf-8") as handle:
            for document_id, document in corpus.items():
                handle.write(json.dumps({"id": document_id, "contents": f"{document.get('title', '')}\n\n{document.get('text', '')}"}) + "\n")
        index = Path(temporary) / "index"
        subprocess.run([sys.executable, "-m", "pyserini.index", "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input", temporary, "-index", str(index), "-storePositions", "-storeDocvectors", "-storeRaw"], check=True)
        from pyserini.search.lucene import LuceneSearcher as SimpleSearcher
        searcher = SimpleSearcher(str(index))
        bm25 = {query_id: [hit.docid for hit in searcher.search(query, k=args.search_depth)] for query_id, query in selected}
        model = SentenceTransformer(args.model)
        document_ids = list(corpus)
        document_texts = [f"{corpus[doc].get('title', '')}\n\n{corpus[doc].get('text', '')}" for doc in document_ids]
        document_vectors = model.encode(document_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        dense = {}
        for query_id, query in selected:
            scores = util.cos_sim(model.encode(query, normalize_embeddings=True), document_vectors)[0]
            top = scores.topk(min(args.search_depth, len(document_ids))).indices.tolist()
            dense[query_id] = [document_ids[index] for index in top]
    fused = rrf(bm25, dense)
    metrics = {}
    for count in counts:
        ids = [query_id for query_id, _ in selected[:count]]
        metrics[str(count)] = evaluate_method(
            {key: qrels[key] for key in ids},
            {key: fused[key] for key in ids},
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": args.dataset, "split": args.split, "sampleSeed": args.sample_seed, "queryIds": [query_id for query_id, _ in selected], "searchDepth": args.search_depth, "model": args.model, "rrfConstant": 60, "method": "local bounded Anserini-style BM25+BGE-small RRF", "documentRankings": fused, "metricsByQueryCount": metrics}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
