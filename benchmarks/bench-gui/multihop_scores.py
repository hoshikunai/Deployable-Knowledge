from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import paths

COMPLETE_CUTOFFS = (2, 5, 10)
DETAIL_CUTOFF = 10
SIDECAR_NAME = "multihop.json"


def load_multihop(dataset: str) -> dict[str, Any] | None:
    path = paths.DATASETS_ROOT / dataset / SIDECAR_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ranked_documents(ranking: dict[str, float], query_id: str) -> list[str]:
    ordered = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    return [document for document, _ in ordered if document != query_id]


def canonical_gold_hops(entry: dict[str, Any], beir_to_application: dict[str, str],
                        application_to_beir: dict[str, str]) -> dict[str, int | None]:
    hops: dict[str, int | None] = {}
    for gold in entry["gold"]:
        canonical = application_to_beir.get(beir_to_application.get(gold["id"], ""), gold["id"])
        hops[canonical] = gold["hop"]
    return hops


def share(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def per_query_complete(rankings: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]],
                       query_ids: list[str], cutoff: int = DETAIL_CUTOFF) -> dict[str, float]:
    completes: dict[str, float] = {}
    for query_id in query_ids:
        gold = {document for document, grade in qrels.get(query_id, {}).items() if grade > 0}
        if not gold or query_id not in rankings:
            continue
        top = set(ranked_documents(rankings[query_id], query_id)[:cutoff])
        completes[query_id] = float(gold <= top)
    return completes


def evaluate_method(rankings: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]],
                    query_ids: list[str], sidecar: dict[str, Any], beir_to_application: dict[str, str],
                    application_to_beir: dict[str, str]) -> dict[str, Any]:
    complete_totals = {cutoff: 0 for cutoff in COMPLETE_CUTOFFS}
    recall_total = 0.0
    evaluated = 0
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"queries": 0, "complete": 0, "recall": 0.0})
    by_hop: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "found": 0})
    unlabelled = 0
    for query_id in query_ids:
        gold = {document for document, grade in qrels.get(query_id, {}).items() if grade > 0}
        entry = sidecar["queries"].get(query_id)
        if not gold or entry is None or query_id not in rankings:
            continue
        evaluated += 1
        ranked = ranked_documents(rankings[query_id], query_id)
        for cutoff in COMPLETE_CUTOFFS:
            complete_totals[cutoff] += gold <= set(ranked[:cutoff])
        top = set(ranked[:DETAIL_CUTOFF])
        recall = len(gold & top) / len(gold)
        recall_total += recall
        group = by_type[entry["type"]]
        group["queries"] += 1
        group["complete"] += gold <= top
        group["recall"] += recall
        for document, hop in canonical_gold_hops(entry, beir_to_application, application_to_beir).items():
            if hop is None:
                unlabelled += 1
                continue
            bucket = by_hop["hop 1" if hop == 1 else "hop 2+"]
            bucket["gold"] += 1
            bucket["found"] += document in top
    return {
        "queries": evaluated,
        "complete": {f"Complete@{cutoff}": share(total, evaluated) for cutoff, total in complete_totals.items()},
        f"Recall@{DETAIL_CUTOFF}": share(recall_total, evaluated),
        "byType": {
            name: {"queries": int(group["queries"]),
                   f"Complete@{DETAIL_CUTOFF}": share(group["complete"], int(group["queries"])),
                   f"Recall@{DETAIL_CUTOFF}": share(group["recall"], int(group["queries"]))}
            for name, group in sorted(by_type.items())
        },
        "byHop": {
            name: {"gold": bucket["gold"], f"Recall@{DETAIL_CUTOFF}": share(bucket["found"], bucket["gold"])}
            for name, bucket in sorted(by_hop.items())
        },
        "goldWithoutHop": unlabelled,
    }


def evaluate_multihop(rankings: dict[str, dict[str, dict[str, float]]], qrels: dict[str, dict[str, int]],
                      query_ids: list[str], sidecar: dict[str, Any], beir_to_application: dict[str, str],
                      application_to_beir: dict[str, str], methods: tuple[str, ...]) -> dict[str, Any]:
    return {
        method: evaluate_method(rankings[method], qrels, query_ids, sidecar,
                                beir_to_application, application_to_beir)
        for method in methods
        if method in rankings
    }
