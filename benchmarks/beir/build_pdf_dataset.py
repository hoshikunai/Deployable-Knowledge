"""Build the pdf-eval BEIR dataset from the app's PDF eval set.

The ranking unit is a page: corpus.jsonl lists one entry per page ("<document>#p<page>"),
and qrels mark every page in each question's first..last range as relevant. run.py uploads
the PDFs through the app's file ingestion, so the selected document parsing applies.

    benchmarks/beir/.venv/bin/python benchmarks/beir/build_pdf_dataset.py \\
        --eval-assets ~/Projects/handoffs/docling/eval-assets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from dataset_catalog import PDF_MANIFEST

HARNESS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = HARNESS_ROOT.parent.parent
DATASET = "pdf-eval"
QUESTION_FILES = ("questions.json", "table-questions.json")

# Mirrors CORPUS in eval-assets/run.mjs: document name -> file in the repository's documents/.
CORPUS = {
    "afi36-2406": "c43e4d361db42702.pdf",
    "aetcman11-251": "3104b32cf4aa2b1a.pdf",
    "dafman36_2806": "057631e71de46bf2.pdf",
    "dafi36-2903": "fd7d9f5c416b1b8e.pdf",
    "afman11-2t-38v3": "5300f0c13308ac27.pdf",
    "afman11-2t-38v2": "f9e8d698ae74f712.pdf",
    "dafi90-160": "2c210c2b2416a960.pdf",
    "BrownBook": "01379f757fa11f89.pdf",
}

PAGE_COUNT_SCRIPT = """
import { readFile } from 'node:fs/promises';
import { PDFParse } from 'pdf-parse';
const counts = {};
for (const path of process.argv.slice(1)) {
	const parser = new PDFParse({ data: await readFile(path) });
	counts[path] = (await parser.getInfo()).total;
	await parser.destroy();
}
console.log(JSON.stringify(counts));
"""


def page_id(document: str, page: int) -> str:
    return f"{document}#p{page}"


def count_pages(files: list[Path]) -> dict[Path, int]:
    """Page counts from pdf-parse, the parser the app's PDF ingestion uses."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PAGE_COUNT_SCRIPT, *map(str, files)],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True,
    )
    counts = json.loads(result.stdout)
    return {path: counts[str(path)] for path in files}


def load_questions(eval_assets: Path) -> list[dict]:
    questions = []
    for name in QUESTION_FILES:
        questions += json.loads((eval_assets / name).read_text(encoding="utf-8"))
    identifiers = [question["id"] for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Question IDs repeat across the question files")
    return questions


def validate_questions(questions: list[dict], pages: dict[str, int]) -> None:
    for question in questions:
        document = question["document"]
        if document not in pages:
            raise ValueError(f"{question['id']} names unknown document {document}")
        if not 1 <= question["first"] <= question["last"] <= pages[document]:
            raise ValueError(f"{question['id']} pages {question['first']}-{question['last']} "
                             f"are outside {document}'s {pages[document]} pages")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def build(eval_assets: Path, output: Path) -> dict:
    sources = {name: REPOSITORY_ROOT / "documents" / file for name, file in CORPUS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Eval PDFs are missing: {', '.join(missing)}")
    questions = load_questions(eval_assets)
    counts = count_pages(list(sources.values()))
    pages = {name: counts[path] for name, path in sources.items()}
    validate_questions(questions, pages)

    (output / "pdfs").mkdir(parents=True)
    (output / "qrels").mkdir()
    documents = []
    for name, source in sources.items():
        relative = f"pdfs/{name}.pdf"
        shutil.copy2(source, output / relative)
        documents.append({"id": name, "file": relative, "pages": pages[name],
                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})

    write_jsonl(output / "corpus.jsonl", [
        {"_id": page_id(name, page), "title": name, "text": ""}
        for name in CORPUS for page in range(1, pages[name] + 1)
    ])
    write_jsonl(output / "queries.jsonl",
                [{"_id": question["id"], "text": question["question"]} for question in questions])
    rows = ["query-id\tcorpus-id\tscore"] + [
        f"{question['id']}\t{page_id(question['document'], page)}\t1"
        for question in questions
        for page in range(question["first"], question["last"] + 1)
    ]
    (output / "qrels" / "test.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = {"unit": "page", "source": "eval-assets " + ", ".join(QUESTION_FILES),
                "documents": documents}
    (output / PDF_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"documents": len(documents), "pages": sum(pages.values()),
            "queries": len(questions), "qrels": len(rows) - 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-assets", type=Path, required=True,
                        help="Folder holding questions.json and table-questions.json.")
    parser.add_argument("--replace", action="store_true",
                        help="Rebuild an existing dataset folder.")
    arguments = parser.parse_args()
    output = HARNESS_ROOT / "datasets" / DATASET
    if output.exists():
        if not arguments.replace:
            raise SystemExit(f"{output} exists; pass --replace to rebuild it")
        shutil.rmtree(output)
    summary = build(arguments.eval_assets.expanduser().resolve(), output)
    print(f"Built {output}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
