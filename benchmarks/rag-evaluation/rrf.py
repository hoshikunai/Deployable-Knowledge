"""Document-level reciprocal-rank fusion for benchmark ablations."""
from __future__ import annotations

def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60, limit: int | None = None) -> list[tuple[str, float]]:
    if k <= 0: raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, 1):
            if not doc_id or doc_id in seen: continue
            seen.add(doc_id); scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    result = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return result if limit is None else result[:limit]
