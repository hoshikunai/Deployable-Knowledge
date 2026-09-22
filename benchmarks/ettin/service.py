"""Loopback Ettin scorer; opt-in and separately launched."""
from __future__ import annotations

import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from sentence_transformers import CrossEncoder

ETTIN_MODELS = {
    "ettin-32m": "cross-encoder/ettin-reranker-32m-v1",
    "ettin-68m": "cross-encoder/ettin-reranker-68m-v1",
    "ettin-150m": "cross-encoder/ettin-reranker-150m-v1",
    "ettin-400m": "cross-encoder/ettin-reranker-400m-v1",
}
VARIANT = os.getenv("RAG_RERANK_MODEL", "ettin-32m")
if VARIANT not in ETTIN_MODELS:
    raise RuntimeError(f"Unsupported Ettin reranker: {VARIANT}")
MODEL = ETTIN_MODELS[VARIANT]
MAX_LENGTH = 512
torch.set_num_threads(int(os.getenv("ETTIN_CPU_THREADS", "2")))
app = FastAPI()
model = CrossEncoder(MODEL, backend="torch", device="cpu", max_length=MAX_LENGTH)


class Candidate(BaseModel):
    chunkId: str
    content: str


class Request(BaseModel):
    query: str
    candidates: list[Candidate]
    maxLength: int = MAX_LENGTH


@app.post("/rerank")
async def rerank(request: Request) -> dict[str, list[float] | str]:
    if (
        not request.query.strip()
        or request.maxLength != MAX_LENGTH
        or not request.candidates
        or len(request.candidates) > 32
    ):
        raise HTTPException(400, "query must be non-empty and maxLength must be 512")
    try:
        scores = model.predict(
            [(request.query, candidate.content) for candidate in request.candidates],
            batch_size=32,
            show_progress_bar=False,
        )
        values = [float(score) for score in scores]
    except Exception as error:
        raise HTTPException(500, f"Ettin inference failed: {error}") from error
    if len(values) != len(request.candidates) or not all(math.isfinite(value) for value in values):
        raise HTTPException(500, "Ettin produced an invalid score vector")
    return {"scores": values, "model": MODEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("ETTIN_PORT", "41792")))
