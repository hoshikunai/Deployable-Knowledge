"""Build a BEIR PDF dataset from OHR-Bench (OCR Hinders RAG, ICCV 2025).

OHR-Bench holds PDFs from seven domains with questions labelled by the evidence page and the
kind of element that holds the answer (text, table, formula, chart, reading order). The default
takes the four domains with the most table questions; --domains changes that. Every PDF of a
chosen domain is included, with or without questions.

Query groups: source is the evidence kind, table is "yes" for table evidence, type is the
domain, and textLayer marks questions whose evidence page is a scan (table recognition skips
those pages).

    benchmarks/beir/.venv/bin/python benchmarks/beir/build_ohr_bench_dataset.py
    benchmarks/beir/.venv/bin/python benchmarks/beir/build_ohr_bench_dataset.py \\
        --name ohr-bench-all --domains all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from dataset_catalog import validate_dataset_name
from pdf_dataset_tools import (PdfDocument, PdfQuestion, check_numbering, document_id, fetch,
                               prepare_output, read_page_texts, text_layer, write_pdf_dataset)

HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
PDFS_URL = "https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/main/pdfs.zip"
QUESTIONS_URL = "https://raw.githubusercontent.com/opendatalab/OHR-Bench/main/data/qas_v2.json"
DOMAINS = ("academic", "administration", "finance", "law", "manual", "news", "textbook")
TABLE_DOMAINS = ("academic", "administration", "finance", "manual")
# evidence_page_no counts from 0; check_numbering confirms it against the PDFs.
PUBLISHED_BASE = 0
ARCHIVE_ENTRY = re.compile(r"^([a-z]+)/([^/\\]+)\.pdf$")


def parse_domains(value: str) -> tuple[str, ...]:
    if value == "all":
        return DOMAINS
    domains = tuple(sorted(set(value.split(","))))
    if unknown := [domain for domain in domains if domain not in DOMAINS]:
        raise argparse.ArgumentTypeError(f"Unknown domains: {', '.join(unknown)}")
    return domains


def extract_pdfs(archive: Path, cache: Path, domains: tuple[str, ...]) -> dict[str, Path]:
    """Extract the chosen domains' PDFs; returns {OHR doc_name: path}."""
    extracted = {}
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if name.endswith("/"):
                continue
            match = ARCHIVE_ENTRY.fullmatch(name)
            if match is None:
                raise RuntimeError(f"Unexpected entry in {archive.name}: {name}")
            if match.group(1) not in domains:
                continue
            target = cache / "pdfs" / f"{document_id(name.removesuffix('.pdf'))}.pdf"
            if target in extracted.values():
                raise RuntimeError(f"{name} maps to the same document ID as another PDF")
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(name))
            extracted[name.removesuffix(".pdf")] = target
    return extracted


def evidence_pages(question: dict) -> list[int]:
    pages = question["evidence_page_no"]
    return pages if isinstance(pages, list) else [pages]


def build(output: Path, cache: Path, domains: tuple[str, ...]) -> dict:
    fetch(PDFS_URL, cache / "pdfs.zip", expected_prefix=b"PK")
    fetch(QUESTIONS_URL, cache / "qas_v2.json")
    sources = extract_pdfs(cache / "pdfs.zip", cache, domains)
    texts = read_page_texts(sorted(sources.values()))
    page_texts = {name: texts[path] for name, path in sources.items()}
    if empty := [name for name, pages in page_texts.items() if not pages]:
        raise RuntimeError(f"pdf-parse read no pages from {', '.join(empty)}")
    ids = {name: document_id(name) for name in sources}

    records = json.loads((cache / "qas_v2.json").read_text(encoding="utf-8"))
    chosen = [record for record in records if record["doc_type"] in domains]
    if missing := sorted({record["doc_name"] for record in chosen} - set(sources)):
        raise RuntimeError(f"Questions name PDFs missing from the archive: {', '.join(missing[:5])}")

    samples = [(record["doc_name"], record["evidence_page_no"], record["evidence_context"])
               for record in chosen
               if isinstance(record["evidence_page_no"], int) and isinstance(record["evidence_context"], str)]
    numbering = check_numbering(samples, page_texts, PUBLISHED_BASE)

    questions = []
    for record in chosen:
        evidence = tuple((record["doc_name"], page - PUBLISHED_BASE + 1) for page in evidence_pages(record))
        questions.append(PdfQuestion(
            id=f"{record['evidence_source']}-{record['ID']}",
            text=record["questions"],
            evidence=tuple((ids[name], page) for name, page in evidence),
            source=record["evidence_source"],
            type=record["doc_type"],
            text_layer=text_layer(evidence, page_texts),
            table=record["evidence_source"] == "table",
            answer={"answer": record["answers"], "answerForm": record["answer_form"],
                    "evidence": record["evidence_context"]},
        ))
    documents = [PdfDocument(ids[name], path, len(page_texts[name]))
                 for name, path in sorted(sources.items())]
    summary = write_pdf_dataset(output, documents, questions, {
        "source": "opendatalab/OHR-Bench pdfs.zip and data/qas_v2.json (research use only)",
        "domains": list(domains),
        "sourceSha256": {name: hashlib.sha256((cache / name).read_bytes()).hexdigest()
                         for name in ("pdfs.zip", "qas_v2.json")},
        "pageNumbering": {"publishedBase": PUBLISHED_BASE, "evidenceMatchRates": numbering},
    })
    return {**summary, "tableQueries": sum(bool(question.table) for question in questions),
            "evidenceMatchRates": numbering}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="ohr-bench", help="Dataset folder name.")
    parser.add_argument("--domains", type=parse_domains, default=TABLE_DOMAINS,
                        help=f"Comma-separated domains or 'all' (default: {','.join(TABLE_DOMAINS)}).")
    parser.add_argument("--source-cache", type=Path, default=DATASETS_ROOT / ".ohr-bench-source",
                        help="Where the downloaded archive, questions and extracted PDFs are kept.")
    parser.add_argument("--replace", action="store_true", help="Rebuild an existing dataset folder.")
    arguments = parser.parse_args()
    output = DATASETS_ROOT / validate_dataset_name(arguments.name)
    prepare_output(output, arguments.replace)
    summary = build(output, arguments.source_cache.expanduser().resolve(), arguments.domains)
    print(f"Built {output}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
