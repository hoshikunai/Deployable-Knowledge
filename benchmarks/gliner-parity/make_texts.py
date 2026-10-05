from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity"
SOURCES = {
    "17-13-tactical-casualty-combat-care-handbook-v5-may-17-distro-a": 12,
    "afi36-2406": 4,
    "dafman36_2806": 4,
}
MIN_WORDS = 60
MAX_WORDS = 300
GENERAL = [
    "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC "
    "interpreters for the Altair 8800.",
    "The Eiffel Tower, located in Paris, France, was designed by engineer Gustave Eiffel and completed in 1889.",
]
ENTITY_LABELS = [
    "person",
    "organization",
    "military role",
    "medical procedure",
    "medication",
    "injury",
    "medical equipment",
    "document",
]
RELATIONS = ["treats", "requires", "used for", "responsible for"]


def sample(database: Path) -> list[dict[str, str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    chunks: list[dict[str, str]] = []
    for title, count in SOURCES.items():
        rows = connection.execute(
            "SELECT dc.id, dc.content FROM document_chunks dc JOIN documents d ON d.id = dc.document_id "
            "WHERE d.title = ? AND dc.chunk_type = 'TEXT' ORDER BY dc.page_index, dc.chunk_index",
            (title,),
        ).fetchall()
        usable = [(chunk_id, content) for chunk_id, content in rows
                  if MIN_WORDS <= len(content.split()) <= MAX_WORDS]
        if len(usable) < count:
            raise ValueError(f"{title}: only {len(usable)} usable chunks, {count} needed")
        step = len(usable) / count
        chunks += [{"id": usable[int(index * step)][0], "source": title, "text": usable[int(index * step)][1]}
                   for index in range(count)]
    chunks += [{"id": f"general-{index}", "source": "general", "text": text} for index, text in enumerate(GENERAL)]
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample fixed GLiNER test texts from an app database.")
    parser.add_argument("--database", type=Path, default=ROOT / "app.db")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {"entityLabels": ENTITY_LABELS, "relations": RELATIONS, "chunks": sample(arguments.database)}
    (OUT / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    words = [len(chunk["text"].split()) for chunk in texts["chunks"]]
    print(f"{len(words)} chunks, {min(words)}-{max(words)} words -> {OUT / 'texts.json'}")


if __name__ == "__main__":
    main()
