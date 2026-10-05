from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import OUT, PREDICTIONS_PATH, SET_PATH, same_entity

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DETAIL_THRESHOLD = 0.7


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record
    return records


def predicted_triples(record: dict[str, Any], threshold: float) -> list[tuple[str, str, str]]:
    triples = {}
    for paragraph in record["paragraphs"]:
        for relation in paragraph["relations"]:
            if relation["score"] > threshold:
                triples[(relation["head"], relation["relation"], relation["tail"])] = None
    return list(triples)


def entity_texts(record: dict[str, Any]) -> list[str]:
    return [entity["text"] for paragraph in record["paragraphs"] for entity in paragraph["entities"]]


def gold_outcome(gold: dict[str, Any], predicted: list[tuple[str, str, str]], lenient: bool) -> str | None:
    same_relation = [triple for triple in predicted if triple[1] == gold["relation"]]
    if any(same_entity(gold["subject"], head, lenient) and same_entity(gold["object"], tail, lenient)
           for head, _, tail in same_relation):
        return "found"
    if any(same_entity(gold["subject"], tail, lenient) and same_entity(gold["object"], head, lenient)
           for head, _, tail in same_relation):
        return "reversed"
    return None


def prediction_correct(triple: tuple[str, str, str], gold: list[dict[str, Any]], lenient: bool) -> bool:
    head, relation, tail = triple
    return any(item["relation"] == relation and same_entity(item["subject"], head, lenient)
               and same_entity(item["object"], tail, lenient) for item in gold)


def diagnose(gold: dict[str, Any], record: dict[str, Any], lenient: bool) -> str:
    outcome = gold_outcome(gold, predicted_triples(record, DETAIL_THRESHOLD), lenient)
    if outcome:
        return outcome
    entities = entity_texts(record)
    subject = any(same_entity(gold["subject"], text, lenient) for text in entities)
    target = any(same_entity(gold["object"], text, lenient) for text in entities)
    if subject and target:
        below = gold_outcome(gold, predicted_triples(record, 0.0), lenient)
        return "below threshold" if below else "relation missed"
    return "entity missed"


def evaluate(evaluation_set: dict[str, Any], predictions: dict[str, dict[str, Any]], lenient: bool) -> dict[str, Any]:
    items = [item for item in evaluation_set["items"] if item["id"] in predictions]
    rows = []
    for threshold in THRESHOLDS:
        found = reversed_count = findable_found = correct = predicted_total = 0
        for item in items:
            predicted = predicted_triples(predictions[item["id"]], threshold)
            predicted_total += len(predicted)
            correct += sum(prediction_correct(triple, item["gold"], lenient) for triple in predicted)
            for gold in item["gold"]:
                outcome = gold_outcome(gold, predicted, lenient)
                found += outcome == "found"
                reversed_count += outcome == "reversed"
                findable_found += outcome == "found" and gold["findable"]
        gold_total = sum(len(item["gold"]) for item in items)
        findable_total = sum(gold["findable"] for item in items for gold in item["gold"])
        rows.append({
            "threshold": threshold,
            "recall": found / gold_total if gold_total else None,
            "findableRecall": findable_found / findable_total if findable_total else None,
            "precision": correct / predicted_total if predicted_total else None,
            "reversed": reversed_count / gold_total if gold_total else None,
            "predictionsPerQuestion": predicted_total / len(items) if items else None,
        })
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    causes: Counter[str] = Counter()
    for item in items:
        record = predictions[item["id"]]
        for gold in item["gold"]:
            cause = diagnose(gold, record, lenient)
            by_relation[gold["relation"]][cause] += 1
            by_type[item["type"]][cause] += 1
            if gold["findable"]:
                causes[cause] += 1
    return {"rows": rows, "byRelation": by_relation, "byType": by_type, "findableCauses": causes}


def percent(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(row) + " |" for row in rows)]


def breakdown(groups: dict[str, Counter[str]]) -> list[list[str]]:
    rows = []
    for name, counts in sorted(groups.items(), key=lambda entry: -sum(entry[1].values())):
        total = sum(counts.values())
        rows.append([name, str(total), percent(counts["found"] / total), percent(counts["reversed"] / total),
                     percent(counts["entity missed"] / total)])
    return rows


def report(evaluation_set: dict[str, Any], predictions: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    scored = sum(item["id"] in predictions for item in evaluation_set["items"])
    seconds = [record["seconds"] for record in predictions.values()]
    strict, lenient = evaluate(evaluation_set, predictions, False), evaluate(evaluation_set, predictions, True)
    lines = ["# Relex on 2WikiMultiHopQA gold paragraphs", "",
             f"{scored} of {len(evaluation_set['items'])} questions scored, "
             f"{evaluation_set['goldTriples']} gold triples in the set, "
             f"{evaluation_set['findableTriples']} with both entities written in the text. "
             f"Median {statistics.median(seconds):.2f}s per question." if seconds else "No predictions yet.", ""]
    for name, result in (("Lenient entity matching", lenient), ("Exact entity matching", strict)):
        lines += [f"## {name}", ""]
        lines += table(["Relation threshold", "Recall", "Recall (findable)", "Precision vs gold", "Reversed",
                        "Triples per question"],
                       [[f"{row['threshold']:.1f}", percent(row["recall"]), percent(row["findableRecall"]),
                         percent(row["precision"]), percent(row["reversed"]),
                         "–" if row["predictionsPerQuestion"] is None else f"{row['predictionsPerQuestion']:.1f}"]
                        for row in result["rows"]])
        lines.append("")
    causes = lenient["findableCauses"]
    total = sum(causes.values())
    lines += [f"## Why findable gold triples were missed (lenient, threshold {DETAIL_THRESHOLD})", ""]
    lines += table(["Outcome", "Share"], [[cause, percent(count / total)] for cause, count in causes.most_common()])
    lines += ["", f"## By relation (lenient, threshold {DETAIL_THRESHOLD})", ""]
    lines += table(["Relation", "Gold", "Found", "Reversed", "Entity missed"], breakdown(lenient["byRelation"]))
    lines += ["", f"## By question type (lenient, threshold {DETAIL_THRESHOLD})", ""]
    lines += table(["Type", "Gold", "Found", "Reversed", "Entity missed"], breakdown(lenient["byType"]))
    lines += ["", "Precision counts only triples in the gold set; paragraphs hold true facts the gold set "
                  "leaves out, so it is a lower bound."]
    summary = {"scoredQuestions": scored, "lenient": lenient, "strict": strict}
    return lines, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score relex predictions against 2Wiki gold evidence triples.")
    parser.add_argument("--set", type=Path, default=SET_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    arguments = parser.parse_args()
    evaluation_set = json.loads(arguments.set.read_text(encoding="utf-8"))
    lines, summary = report(evaluation_set, load_predictions(arguments.predictions))
    markdown = "\n".join(lines) + "\n"
    (OUT / "2wiki-results.md").write_text(markdown, encoding="utf-8")
    (OUT / "2wiki-summary.json").write_text(json.dumps(summary, indent=2, default=dict) + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
