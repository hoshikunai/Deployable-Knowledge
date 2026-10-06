from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from common import OUT, ROOT

OWN_CHUNKS_PATH = OUT / "own-chunks.json"
SOURCES = {
    "17-13-tactical-casualty-combat-care-handbook-v5-may-17-distro-a": 8,
    "aetcman11-251": 6,
    "afi36-2406": 5,
    "dafman36_2806": 5,
    "dafi36-2903": 3,
    "afman11-2t-38v2": 3,
}
MIN_WORDS = 80
MAX_WORDS = 260
LEADER_DOTS = re.compile(r"\.{5,}")


def is_prose(text: str) -> bool:
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS or LEADER_DOTS.search(text):
        return False
    letters = sum(character.isalpha() for character in text)
    return letters / max(1, len(text)) > 0.65


def sample(database: Path) -> list[dict[str, str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    chunks: list[dict[str, str]] = []
    for title, count in SOURCES.items():
        rows = connection.execute(
            "SELECT dc.id, dc.content FROM document_chunks dc JOIN documents d ON d.id = dc.document_id "
            "WHERE d.title = ? AND dc.chunk_type = 'TEXT' ORDER BY dc.page_index, dc.chunk_index",
            (title,),
        ).fetchall()
        prose = [(chunk_id, content) for chunk_id, content in rows if is_prose(content)]
        if len(prose) < count:
            raise ValueError(f"{title}: only {len(prose)} prose chunks, {count} needed")
        step = len(prose) / count
        for index in range(count):
            chunk_id, content = prose[int((index + 0.5) * step)]
            chunks.append({"id": chunk_id, "source": title, "text": content})
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample prose chunks from the app database for a relex check.")
    parser.add_argument("--database", type=Path, default=ROOT / "app.db")
    arguments = parser.parse_args()
    chunks = sample(arguments.database)
    OUT.mkdir(parents=True, exist_ok=True)
    OWN_CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks -> {OWN_CHUNKS_PATH}")


if __name__ == "__main__":
    main()
