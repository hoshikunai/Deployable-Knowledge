"""Check the app's embeddings against the official sentence-transformers implementation.

    RAG_EMBEDDING_MODEL=<id> node_modules/.bin/tsx benchmarks/embedding-parity/embed.mts app.json
    benchmarks/beir/.venv/bin/python benchmarks/embedding-parity/compare.py app.json

Passes when mean cosine similarity is at least 0.99, the minimum at least 0.98, and every query
ranks the same top-5 passages in the same order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# App model ID -> official weights and the prompts the app uses
REFERENCES = {
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: ", True),
    "bekko-embedding-v1-a25m": ("hotchpotch/bekko-embedding-v1-a25m", "", "", False),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5",
                         "Represent this sentence for searching relevant passages: ", "", False),
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5",
                          "Represent this sentence for searching relevant passages: ", "", False),
    "granite-embedding-97m-multilingual-r2": ("ibm-granite/granite-embedding-97m-multilingual-r2",
                                              "", "", False),
    "granite-embedding-english-r2": ("ibm-granite/granite-embedding-english-r2", "", "", False),
    "jina-embeddings-v5-text-nano-retrieval": ("jinaai/jina-embeddings-v5-text-nano-retrieval",
                                               "Query: ", "Document: ", True),
}
MEAN_COSINE, MIN_COSINE, TOP = 0.99, 0.98, 5


def main(path: str) -> int:
    app = json.loads(Path(path).read_text())
    texts = json.loads((Path(__file__).parent / "texts.json").read_text())
    repository, query_prompt, document_prompt, remote_code = REFERENCES[app["model"]]
    # Full precision even when the repository's config defaults to bfloat16 (Jina v5)
    model = SentenceTransformer(repository, device="cpu", trust_remote_code=remote_code,
                                model_kwargs={"dtype": torch.float32})
    reference_queries = model.encode([query_prompt + t for t in texts["queries"]], normalize_embeddings=True)
    reference_passages = model.encode([document_prompt + t for t in texts["passages"]],
                                      normalize_embeddings=True)
    app_queries, app_passages = np.array(app["queries"]), np.array(app["passages"])

    cosines = np.concatenate([
        np.sum(app_queries * reference_queries, axis=1),
        np.sum(app_passages * reference_passages, axis=1),
    ])
    app_top = np.argsort(-(app_queries @ app_passages.T), axis=1)[:, :TOP]
    reference_top = np.argsort(-(reference_queries @ reference_passages.T), axis=1)[:, :TOP]
    same_top = [bool((a == r).all()) for a, r in zip(app_top, reference_top)]

    passed = cosines.mean() >= MEAN_COSINE and cosines.min() >= MIN_COSINE and all(same_top)
    print(f"{app['model']}: dims {app_queries.shape[1]} vs {reference_queries.shape[1]} | "
          f"cosine mean {cosines.mean():.4f} min {cosines.min():.4f} | "
          f"same top-{TOP}: {sum(same_top)}/{len(same_top)} queries | "
          f"app embed time {app['seconds']:.2f}s | {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
