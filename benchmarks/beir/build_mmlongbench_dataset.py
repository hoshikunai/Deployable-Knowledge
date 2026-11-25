"""Build a BEIR PDF dataset from MMLongBench-Doc (NeurIPS 2024 Datasets and Benchmarks).

135 long PDFs (reports, papers, guidebooks, slides, brochures, financial reports) with questions
labelled by evidence pages and evidence kinds (text, layout, table, chart, figure). Questions
without evidence pages are unanswerable by design and are left out, as are the few whose labels
cite pages the PDF does not have and those on files that duplicate another file byte for byte.
pdf-manifest.json lists every question left out.

Query groups: source is the evidence kinds joined with "+", table is "yes" when they include a
table, type is the document type, and textLayer marks questions whose evidence page is a scan.

    benchmarks/beir/.venv/bin/python benchmarks/beir/build_mmlongbench_dataset.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import urllib.parse
from pathlib import Path

from dataset_catalog import validate_dataset_name
from pdf_dataset_tools import (PdfDocument, PdfQuestion, check_numbering, document_id, fetch,
                               prepare_output, read_page_texts, text_layer, write_pdf_dataset)

HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
QUESTIONS_URL = "https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/data/samples.json"
DOCUMENTS_URL = "https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/resolve/main/documents"
EVIDENCE_KINDS = {"Pure-text (Plain-text)": "text", "Generalized-text (Layout)": "layout",
                  "Table": "table", "Chart": "chart", "Figure": "figure"}
# evidence_pages count from 1; check_numbering confirms it against the PDFs.
PUBLISHED_BASE = 1
# Answers of at least this many words are specific enough to locate their page.
MIN_SAMPLE_ANSWER_WORDS = 3


def parse_list(value: str | list, label: str) -> list:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise RuntimeError(f"Malformed {label}: {value!r}")
    return parsed


def evidence_kinds(record: dict) -> str:
    kinds = parse_list(record["evidence_sources"], "evidence_sources")
    if unknown := [kind for kind in kinds if kind not in EVIDENCE_KINDS]:
        raise RuntimeError(f"Unknown evidence sources: {unknown}")
    return "+".join(sorted({EVIDENCE_KINDS[kind] for kind in kinds})) or "unlabelled"


def build(output: Path, cache: Path) -> dict:
    fetch(QUESTIONS_URL, cache / "samples.json")
    records = json.loads((cache / "samples.json").read_text(encoding="utf-8"))
    answerable = [(index, record) for index, record in enumerate(records)
                  if parse_list(record["evidence_pages"], "evidence_pages")]
    unanswerable = len(records) - len(answerable)
    names = sorted({record["doc_id"] for record in records})
    for position, name in enumerate(names, start=1):
        if fetch(f"{DOCUMENTS_URL}/{urllib.parse.quote(name)}", cache / "pdfs" / name,
                 expected_prefix=b"%PDF-"):
            print(f"Downloaded {position}/{len(names)} documents")
    # Some files in the source are byte-identical under different names, and at least two of
    # them hold another document than their name and questions describe. The first name of each
    # identical group is kept; questions on the others are left out.
    first_by_hash: dict[str, str] = {}
    duplicates = {}
    for name in names:
        digest = hashlib.sha256((cache / "pdfs" / name).read_bytes()).hexdigest()
        if digest in first_by_hash:
            duplicates[name] = first_by_hash[digest]
        else:
            first_by_hash[digest] = name
    names = [name for name in names if name not in duplicates]
    duplicate_questions = [index for index, record in answerable if record["doc_id"] in duplicates]
    answerable = [(index, record) for index, record in answerable if record["doc_id"] not in duplicates]
    sources = {name: cache / "pdfs" / name for name in names}
    texts = read_page_texts(list(sources.values()))
    page_texts = {name: texts[path] for name, path in sources.items()}
    if empty := [name for name, pages in page_texts.items() if not pages]:
        raise RuntimeError(f"pdf-parse read no pages from {', '.join(empty)}")
    ids = {name: document_id(name.removesuffix(".pdf")) for name in names}
    # A few published labels cite page 0 or pages past the end of the PDF (printed page numbers,
    # typos); those questions cannot be scored and are left out.
    impossible = [index for index, record in answerable
                  if any(not 1 <= page <= len(page_texts[record["doc_id"]])
                         for page in parse_list(record["evidence_pages"], "evidence_pages"))]
    answerable = [(index, record) for index, record in answerable if index not in impossible]

    samples = []
    for _, record in answerable:
        pages = parse_list(record["evidence_pages"], "evidence_pages")
        if (len(pages) == 1 and evidence_kinds(record) == "text"
                and len(str(record["answer"]).split()) >= MIN_SAMPLE_ANSWER_WORDS):
            samples.append((record["doc_id"], pages[0], str(record["answer"])))
    numbering = check_numbering(samples, page_texts, PUBLISHED_BASE)

    questions = []
    for index, record in answerable:
        pages = tuple(sorted({(record["doc_id"], page - PUBLISHED_BASE + 1)
                              for page in parse_list(record["evidence_pages"], "evidence_pages")}))
        questions.append(PdfQuestion(
            id=f"mmlongbench-{index:04d}",
            text=record["question"],
            evidence=tuple((ids[name], page) for name, page in pages),
            source=evidence_kinds(record),
            type=record["doc_type"],
            text_layer=text_layer(pages, page_texts),
            table="table" in evidence_kinds(record).split("+"),
            answer={"answer": record["answer"], "answerFormat": record["answer_format"]},
        ))
    documents = [PdfDocument(ids[name], sources[name], len(page_texts[name])) for name in names]
    summary = write_pdf_dataset(output, documents, questions, {
        "source": "mayubo2333/MMLongBench-Doc data/samples.json and yubo2333/MMLongBench-Doc "
                  "documents/ (research use, CC-BY-NC-4.0)",
        "sourceSha256": {"samples.json": hashlib.sha256((cache / "samples.json").read_bytes()).hexdigest()},
        "unanswerableQuestionsLeftOut": unanswerable,
        "impossibleEvidenceQuestionsLeftOut": [f"mmlongbench-{index:04d}" for index in impossible],
        "duplicateFilesLeftOut": duplicates,
        "duplicateFileQuestionsLeftOut": [f"mmlongbench-{index:04d}" for index in duplicate_questions],
        "pageNumbering": {"publishedBase": PUBLISHED_BASE, "evidenceMatchRates": numbering,
                          "samples": len(samples)},
    })
    return {**summary, "tableQueries": sum(bool(question.table) for question in questions),
            "evidenceMatchRates": numbering, "numberingSamples": len(samples),
            "impossibleEvidenceQuestions": len(impossible),
            "duplicateFileQuestions": len(duplicate_questions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="mmlongbench-doc", help="Dataset folder name.")
    parser.add_argument("--source-cache", type=Path, default=DATASETS_ROOT / ".mmlongbench-source",
                        help="Where the downloaded questions and PDFs are kept.")
    parser.add_argument("--replace", action="store_true", help="Rebuild an existing dataset folder.")
    arguments = parser.parse_args()
    output = DATASETS_ROOT / validate_dataset_name(arguments.name)
    prepare_output(output, arguments.replace)
    summary = build(output, arguments.source_cache.expanduser().resolve())
    print(f"Built {output}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
