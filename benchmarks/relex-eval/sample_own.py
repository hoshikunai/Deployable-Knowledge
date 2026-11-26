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


def excluded_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            ids.update(chunk["id"] for chunk in data)
        elif "sampleChunkIds" in data:
            ids.update(data["sampleChunkIds"])
        elif "entries" in data:
            ids.update(chunk_id for entry in data["entries"] for chunk_id in entry["sampleChunkIds"])
        else:
            raise ValueError(f"{path}: no chunk list, sampleChunkIds or dictionary entries")
    return ids


def pick(prose: list[tuple[str, str]], position: int, excluded: set[str]) -> tuple[str, str]:
    for chunk_id, content in prose[position:] + prose[:position]:
        if chunk_id not in excluded:
            return chunk_id, content
    raise ValueError("every prose chunk is excluded")


def sample(database: Path, offset: float, excluded: set[str]) -> list[dict[str, str]]:
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
            chunk_id, content = pick(prose, int((index + offset) * step), excluded)
            excluded.add(chunk_id)
            chunks.append({"id": chunk_id, "source": title, "text": content})
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample prose chunks from the app database for a relex check.")
    parser.add_argument("--database", type=Path, default=ROOT / "app.db")
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=OWN_CHUNKS_PATH)
    arguments = parser.parse_args()
    excluded = excluded_ids(arguments.exclude)
    chunks = sample(arguments.database, arguments.offset, excluded)
    OUT.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks, {len(excluded) - len(chunks)} excluded -> {arguments.out}")


if __name__ == "__main__":
    main()
