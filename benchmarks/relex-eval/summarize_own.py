from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import OUT

JUDGEMENTS_PATH = OUT / "own-judgements.json"
RESULTS_PATH = OUT / "own-results.md"
VERDICTS = ("correct", "loose", "wrong", "structural", "duplicate")
SHORT_SOURCES = {
    "17-13-tactical-casualty-combat-care-handbook-v5-may-17-distro-a": "TCCC handbook",
    "aetcman11-251": "AETCMAN 11-251",
    "afi36-2406": "AFI 36-2406",
    "dafman36_2806": "DAFMAN 36-2806",
    "dafi36-2903": "DAFI 36-2903",
    "afman11-2t-38v2": "AFMAN 11-2T-38V2",
}


def percent(part: int, whole: int) -> str:
    return "–" if whole == 0 else f"{part / whole * 100:.0f}%"


def row(name: str, counts: Counter[str]) -> list[str]:
    total = sum(counts.values())
    unique = total - counts["duplicate"]
    return [name, str(total), *(str(counts[verdict]) for verdict in VERDICTS),
            percent(counts["correct"], unique), percent(counts["correct"] + counts["loose"], unique)]


def table(groups: dict[str, Counter[str]], first: str) -> list[str]:
    headers = [first, "Triples", *VERDICTS, "Correct (unique)", "Correct + loose (unique)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row(name, counts)) + " |"
              for name, counts in sorted(groups.items(), key=lambda entry: -sum(entry[1].values()))]
    return lines


def group(judgements: list[dict[str, Any]], key) -> dict[str, Counter[str]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for judgement in judgements:
        groups[key(judgement)][judgement["verdict"]] += 1
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Tally judged relex relations.")
    parser.add_argument("--judgements", type=Path, default=JUDGEMENTS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--title", default="Relex on the document library: judged relations")
    arguments = parser.parse_args()
    judgements = json.loads(arguments.judgements.read_text(encoding="utf-8"))
    unknown = {judgement["verdict"] for judgement in judgements} - set(VERDICTS)
    if unknown:
        raise ValueError(f"Unknown verdicts: {sorted(unknown)}")
    lines = [f"# {arguments.title}", "",
             f"{len(judgements)} relations from 30 chunks, judged against the chunk text. "
             "'Unique' leaves out duplicates, which name resolution would merge.", ""]
    lines += ["## Overall", ""] + table({"All": Counter(j["verdict"] for j in judgements)}, "Set")
    lines += ["", "## By relation label", ""] + table(group(judgements, lambda j: j["relation"]), "Relation")
    lines += ["", "## By document", ""] + table(
        group(judgements, lambda j: SHORT_SOURCES.get(j["source"], j["source"])), "Document")
    lines += ["", "## By relation score", ""] + table(
        group(judgements, lambda j: "0.9 and above" if j["score"] >= 0.9 else
              ("0.8 to 0.9" if j["score"] >= 0.8 else "0.7 to 0.8")), "Score")
    markdown = "\n".join(lines) + "\n"
    arguments.out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
