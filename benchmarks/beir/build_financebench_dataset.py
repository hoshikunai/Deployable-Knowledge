"""Build a BEIR PDF dataset from FinanceBench's open-source sample (Patronus AI).

150 questions over public companies' 10-K, 10-Q and 8-K filings. Each question's evidence
names a filing and a zero-based page; most answers come from financial statement tables. The
corpus is the filings the evidence cites.

Query groups: source is FinanceBench's question type, type its reasoning label, textLayer marks
questions whose evidence page is a scan. FinanceBench does not label table evidence.

    benchmarks/beir/.venv/bin/python benchmarks/beir/build_financebench_dataset.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from dataset_catalog import validate_dataset_name
from pdf_dataset_tools import (PdfDocument, PdfQuestion, check_numbering, document_id, fetch,
                               prepare_output, read_page_texts, text_layer, write_pdf_dataset)

HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
SOURCE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS = "financebench_open_source.jsonl"
# evidence_page_num counts from 0; check_numbering confirms it against the PDFs.
PUBLISHED_BASE = 0


def parse_evidence(record: dict) -> list[dict]:
    """FinanceBench stores each question's evidence as a Python literal list of dicts."""
    evidence = record["evidence"]
    if isinstance(evidence, str):
        evidence = ast.literal_eval(evidence)
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, dict) and isinstance(item.get("doc_name"), str)
        and isinstance(item.get("evidence_page_num"), int)
        and isinstance(item.get("evidence_text_full_page"), str)
        for item in evidence
    ):
        raise RuntimeError(f"Malformed evidence for {record['financebench_id']}")
    return evidence


def build(output: Path, cache: Path) -> dict:
    fetch(f"{SOURCE_URL}/data/{QUESTIONS}", cache / QUESTIONS)
    records = [json.loads(line)
               for line in (cache / QUESTIONS).read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence = {record["financebench_id"]: parse_evidence(record) for record in records}
    filings = sorted({item["doc_name"] for items in evidence.values() for item in items})
    for index, filing in enumerate(filings, start=1):
        if fetch(f"{SOURCE_URL}/pdfs/{filing}.pdf", cache / "pdfs" / f"{filing}.pdf",
                 expected_prefix=b"%PDF-"):
            print(f"Downloaded {index}/{len(filings)} filings")
    sources = {filing: cache / "pdfs" / f"{filing}.pdf" for filing in filings}
    texts = read_page_texts(list(sources.values()))
    page_texts = {filing: texts[path] for filing, path in sources.items()}
    if empty := [filing for filing, pages in page_texts.items() if not pages]:
        raise RuntimeError(f"pdf-parse read no pages from {', '.join(empty)}")
    ids = {filing: document_id(filing) for filing in filings}

    samples = [(item["doc_name"], item["evidence_page_num"], item["evidence_text_full_page"])
               for items in evidence.values() for item in items]
    numbering = check_numbering(samples, page_texts, PUBLISHED_BASE)

    questions = []
    for record in records:
        pages = tuple(sorted({(item["doc_name"], item["evidence_page_num"] - PUBLISHED_BASE + 1)
                              for item in evidence[record["financebench_id"]]}))
        questions.append(PdfQuestion(
            id=record["financebench_id"],
            text=record["question"],
            evidence=tuple((ids[filing], page) for filing, page in pages),
            source=record["question_type"],
            type=record["question_reasoning"] or "unlabelled",
            text_layer=text_layer(pages, page_texts),
            table=None,
            answer={"answer": record["answer"], "justification": record["justification"],
                    "evidence": [item.get("evidence_text", "") for item in evidence[record["financebench_id"]]]},
        ))
    documents = [PdfDocument(ids[filing], sources[filing], len(page_texts[filing])) for filing in filings]
    summary = write_pdf_dataset(output, documents, questions, {
        "source": f"patronus-ai/financebench data/{QUESTIONS} and pdfs/ (CC-BY-NC-4.0)",
        "sourceSha256": {QUESTIONS: hashlib.sha256((cache / QUESTIONS).read_bytes()).hexdigest()},
        "pageNumbering": {"publishedBase": PUBLISHED_BASE, "evidenceMatchRates": numbering},
    })
    return {**summary, "evidenceMatchRates": numbering}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="financebench", help="Dataset folder name.")
    parser.add_argument("--source-cache", type=Path, default=DATASETS_ROOT / ".financebench-source",
                        help="Where the downloaded questions and filings are kept.")
    parser.add_argument("--replace", action="store_true", help="Rebuild an existing dataset folder.")
    arguments = parser.parse_args()
    output = DATASETS_ROOT / validate_dataset_name(arguments.name)
    prepare_output(output, arguments.replace)
    summary = build(output, arguments.source_cache.expanduser().resolve())
    print(f"Built {output}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
