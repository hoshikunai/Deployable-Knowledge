"""Build scifact-dup: SciFact with duplicated relevant documents, for the dedup experiment.

For a seeded half of the documents judged relevant in the test qrels, two copies are added
under new IDs, each carrying the original's relevance label:

- revision: identical text plus a short note at the end. Earlier chunks stay byte-identical
  (what exact dedup targets); the last chunk becomes a near-duplicate.
- edit: one word changed mid-text plus a different note, so the edited chunk is only a
  near-duplicate.

Copies must differ from the original somewhere, because ingestion merges byte-identical
documents into one. duplicate-groups.json maps every copy to its original so the analysis
can count distinct answers.

    benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/make_scifact_dup.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

BEIR_ROOT = Path(__file__).resolve().parents[1]
SOURCE = BEIR_ROOT / "datasets" / "scifact"
TARGET = BEIR_ROOT / "datasets" / "scifact-dup"
SEED = 13
DUPLICATED_SHARE = 0.5
REVISION_NOTE = "Revised version of this record."
EDIT_NOTE = "Copy stored in a second library folder."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def edit_one_word(text: str, generator: random.Random) -> str:
    """Change one word of at least six letters near the middle of the text."""
    words = text.split(" ")
    middle = len(words) // 2
    candidates = sorted((index for index, word in enumerate(words) if word.isalpha() and len(word) >= 6),
                        key=lambda index: abs(index - middle))
    if not candidates:
        return text + " (edited)"
    index = candidates[generator.randrange(min(3, len(candidates)))]
    words[index] = words[index][::-1]
    return " ".join(words)


def main() -> None:
    corpus = read_jsonl(SOURCE / "corpus.jsonl")
    with (SOURCE / "qrels" / "test.tsv").open(encoding="utf-8", newline="") as stream:
        qrels = list(csv.DictReader(stream, delimiter="\t"))

    relevant = sorted({row["corpus-id"] for row in qrels if int(row["score"]) > 0})
    generator = random.Random(SEED)
    chosen = set(generator.sample(relevant, round(len(relevant) * DUPLICATED_SHARE)))
    by_id = {row["_id"]: row for row in corpus}

    copies: list[dict] = []
    groups: dict[str, dict[str, str]] = {}
    for original_id in sorted(chosen):
        original = by_id[original_id]
        variants = {
            "revision": f"{original['text']} {REVISION_NOTE}",
            "edit": f"{edit_one_word(original['text'], generator)} {EDIT_NOTE}",
        }
        for kind, text in variants.items():
            copy_id = f"{original_id}-dup-{kind}"
            copies.append({"_id": copy_id, "title": original.get("title", ""), "text": text,
                           "metadata": {}})
            groups[copy_id] = {"original": original_id, "kind": kind}

    copied_qrels = [
        {**row, "corpus-id": copy_id}
        for row in qrels
        for copy_id, group in groups.items()
        if group["original"] == row["corpus-id"]
    ]

    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "qrels").mkdir(parents=True)
    write_jsonl(TARGET / "corpus.jsonl", corpus + copies)
    shutil.copy2(SOURCE / "queries.jsonl", TARGET / "queries.jsonl")
    with (TARGET / "qrels" / "test.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in ("query-id", "corpus-id", "score")}
                          for row in qrels + copied_qrels])
    (TARGET / "duplicate-groups.json").write_text(json.dumps({
        "seed": SEED, "duplicatedShare": DUPLICATED_SHARE,
        "relevantDocuments": len(relevant), "duplicatedDocuments": len(chosen),
        "copies": groups,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"scifact-dup: {len(corpus)} + {len(copies)} copies of {len(chosen)} of "
          f"{len(relevant)} relevant documents; {len(qrels)} + {len(copied_qrels)} judgments")


if __name__ == "__main__":
    main()
