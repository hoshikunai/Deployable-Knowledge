from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import SET_PATH, mentioned

DEFAULT_ENTITY_LABELS = ("person", "film", "organization", "location", "country", "date", "award", "other")


class SourceFormatError(ValueError):
    pass


def require(record: dict[str, Any], key: str, kind: type, where: str) -> Any:
    if key not in record:
        raise SourceFormatError(f"{where}: missing '{key}'")
    if not isinstance(record[key], kind):
        raise SourceFormatError(f"{where}: '{key}' has type {type(record[key]).__name__}")
    return record[key]


def join_sentences(sentences: list[Any], where: str) -> str:
    if not all(isinstance(sentence, str) for sentence in sentences):
        raise SourceFormatError(f"{where}: sentences must be strings")
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def read_context(record: dict[str, Any], where: str) -> dict[str, str]:
    paragraphs: dict[str, str] = {}
    for position, entry in enumerate(require(record, "context", list, where)):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) \
                or not isinstance(entry[1], list):
            raise SourceFormatError(f"{where}: context[{position}] must be [title, [sentences]]")
        paragraphs.setdefault(entry[0], join_sentences(entry[1], f"{where}: context[{position}]"))
    return paragraphs


def read_triples(record: dict[str, Any], where: str) -> list[list[str]]:
    triples = []
    for position, triple in enumerate(require(record, "evidences", list, where)):
        if not isinstance(triple, list) or len(triple) != 3 or not all(isinstance(part, str) for part in triple):
            raise SourceFormatError(f"{where}: evidences[{position}] must be three strings")
        triples.append([part.strip() for part in triple])
    return triples


def load_items(records: Any) -> tuple[list[dict[str, Any]], Counter[str]]:
    if not isinstance(records, list):
        raise SourceFormatError("source: expected a JSON array of questions")
    items: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise SourceFormatError(f"record {index}: expected an object")
        question_id = require(raw, "_id", str, f"record {index}")
        where = f"question {question_id}"
        context = read_context(raw, where)
        gold_titles: list[str] = []
        for position, fact in enumerate(require(raw, "supporting_facts", list, where)):
            if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str):
                raise SourceFormatError(f"{where}: supporting_facts[{position}] must be [title, index]")
            if fact[0] not in context:
                raise SourceFormatError(f"{where}: supporting title {fact[0]!r} is not in the context")
            if fact[0] not in gold_titles:
                gold_titles.append(fact[0])
        triples = read_triples(raw, where)
        if not triples:
            dropped["no-evidences"] += 1
            continue
        paragraphs = [{"title": title, "text": context[title]} for title in gold_titles]
        text = "\n\n".join(paragraph["text"] for paragraph in paragraphs)
        items.append({
            "id": question_id,
            "type": require(raw, "type", str, where),
            "paragraphs": paragraphs,
            "gold": [{"subject": s, "relation": r, "object": o,
                      "findable": mentioned(s, text) and mentioned(o, text)} for s, r, o in triples],
        })
    return items, dropped


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {name: total * count / population for name, count in counts.items()}
    result = {name: int(value) for name, value in exact.items()}
    for name in sorted(exact, key=lambda name: (result[name] - exact[name], name))[:total - sum(result.values())]:
        result[name] += 1
    return result


def sample(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("--count must be at least 1")
    if len(items) <= count:
        return sorted(items, key=lambda item: item["id"])
    generator = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda item: item["id"]):
        by_type[item["type"]].append(item)
    for name in sorted(by_type):
        generator.shuffle(by_type[name])
    shares = quotas({name: len(group) for name, group in by_type.items()}, count)
    chosen = [item for name in sorted(by_type) for item in by_type[name][:shares[name]]]
    generator.shuffle(chosen)
    return chosen


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, count: int, seed: int, entity_labels: list[str], target: Path = SET_PATH) -> dict[str, Any]:
    items, dropped = load_items(json.loads(source.read_text(encoding="utf-8")))
    chosen = sample(items, count, seed)
    relations = Counter(triple["relation"] for item in chosen for triple in item["gold"])
    gold = [triple for item in chosen for triple in item["gold"]]
    evaluation_set = {
        "source": {"file": source.name, "sha256": file_sha256(source)},
        "seed": seed,
        "entityLabels": entity_labels,
        "relationLabels": [name for name, _ in relations.most_common()],
        "relationCounts": dict(relations.most_common()),
        "typeCounts": dict(sorted(Counter(item["type"] for item in chosen).items())),
        "droppedQuestions": dict(dropped),
        "goldTriples": len(gold),
        "findableTriples": sum(triple["findable"] for triple in gold),
        "items": chosen,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evaluation_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evaluation_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 2WikiMultiHopQA questions with gold evidence triples.")
    parser.add_argument("--source", required=True, type=Path, help="2WikiMultiHopQA dev.json")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entity-labels", default=",".join(DEFAULT_ENTITY_LABELS))
    arguments = parser.parse_args()
    labels = [label.strip() for label in arguments.entity_labels.split(",") if label.strip()]
    result = build(arguments.source, arguments.count, arguments.seed, labels)
    paragraphs = sum(len(item["paragraphs"]) for item in result["items"])
    print(f"{len(result['items'])} questions, {paragraphs} paragraphs, {result['goldTriples']} gold triples "
          f"({result['findableTriples']} findable in the text) -> {SET_PATH}")
    print("relations:", ", ".join(f"{name} ({count})" for name, count in result["relationCounts"].items()))


if __name__ == "__main__":
    main()
