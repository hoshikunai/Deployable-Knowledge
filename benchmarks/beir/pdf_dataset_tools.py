"""Shared pieces of the PDF dataset builders: downloads, pdf-parse page text, the
page-numbering check and the BEIR PDF dataset writer.

A built PDF dataset scores the "test" split by page ("<document>#p<page>") and the "doc" split
by whole document. run.py uploads the PDFs through the app's file ingestion, so the selected
document parsing applies.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from build_pdf_dataset import REPOSITORY_ROOT, page_id, write_jsonl
from dataset_catalog import PDF_MANIFEST

USER_AGENT = "beir-harness-pdf-builder/1.0"
DOWNLOAD_CHUNK_BYTES = 1 << 20
PAGE_TEXT_BATCH = 16
DOCUMENT_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
WORD = re.compile(r"[a-z0-9]{2,}")
EVIDENCE_NGRAM = 3
EVIDENCE_HIT_SHARE = 0.5
# A published page numbering is rejected when the other numbering matches clearly more questions.
NUMBERING_MARGIN = 0.1

PAGE_TEXT_SCRIPT = """
import { readFile } from 'node:fs/promises';
import { PDFParse } from 'pdf-parse';
const texts = {};
for (const path of process.argv.slice(1)) {
	const parser = new PDFParse({ data: await readFile(path) });
	const result = await parser.getText();
	const pages = [];
	for (let page = 1; page <= result.total; page += 1) pages.push(result.getPageText(page));
	texts[path] = pages;
	await parser.destroy();
}
console.log(JSON.stringify(texts));
"""


@dataclass(frozen=True)
class PdfDocument:
    id: str
    source: Path
    pages: int


@dataclass(frozen=True)
class PdfQuestion:
    id: str
    text: str
    # (document id, 1-based page) for every page the dataset marks as evidence.
    evidence: tuple[tuple[str, int], ...]
    source: str
    type: str
    text_layer: str
    # Whether the evidence is a table; None when the dataset does not label it.
    table: bool | None
    answer: dict


def fetch(url: str, destination: Path, expected_prefix: bytes | None = None) -> bool:
    """Stream url to destination unless it is cached. Returns whether it hit the network."""
    if destination.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        first = True
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            if first and expected_prefix is not None and not chunk.startswith(expected_prefix):
                raise RuntimeError(f"{url} did not return the expected file type")
            first = False
            output.write(chunk)
    partial.replace(destination)
    return True


def read_page_texts(files: list[Path]) -> dict[Path, list[str]]:
    """Per-page text from pdf-parse, the parser the app's PDF ingestion uses."""
    texts: dict[Path, list[str]] = {}
    for start in range(0, len(files), PAGE_TEXT_BATCH):
        batch = files[start:start + PAGE_TEXT_BATCH]
        result = subprocess.run(
            ["node", "--input-type=module", "-e", PAGE_TEXT_SCRIPT, *map(str, batch)],
            cwd=REPOSITORY_ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        parsed = json.loads(result.stdout)
        texts.update({path: parsed[str(path)] for path in batch})
    return texts


def document_id(name: str) -> str:
    """A corpus-safe document ID; page IDs append "#p<page>" to it."""
    safe = DOCUMENT_ID_UNSAFE.sub("-", name).strip("-")
    if not safe:
        raise ValueError(f"Cannot derive a document ID from {name!r}")
    return safe


def evidence_ngrams(text: str) -> set[tuple[str, ...]]:
    words = WORD.findall(unicodedata.normalize("NFKC", text).lower())
    return {tuple(words[index:index + EVIDENCE_NGRAM])
            for index in range(len(words) - EVIDENCE_NGRAM + 1)}


def evidence_on_page(evidence: str, page_text: str) -> bool:
    """Whether most of the evidence's word trigrams appear in the page text."""
    wanted = evidence_ngrams(evidence)
    return bool(wanted) and len(wanted & evidence_ngrams(page_text)) >= EVIDENCE_HIT_SHARE * len(wanted)


def numbering_match_rates(samples: list[tuple[str, int, str]], page_texts: dict[str, list[str]],
                          bases: tuple[int, ...] = (0, 1)) -> dict[int, float]:
    """
    Share of (document, published page number, evidence text) samples whose evidence is found on
    the page under each numbering base: base 0 means page number 0 is the first page.
    """
    rates = {}
    for base in bases:
        hits = 0
        for document, number, evidence in samples:
            index = number - base
            pages = page_texts[document]
            hits += 0 <= index < len(pages) and evidence_on_page(evidence, pages[index])
        rates[base] = hits / len(samples) if samples else 0.0
    return rates


def check_numbering(samples: list[tuple[str, int, str]], page_texts: dict[str, list[str]],
                    published_base: int) -> dict[str, float]:
    """Refuse a dataset whose published page numbering disagrees with the PDFs."""
    rates = numbering_match_rates(samples, page_texts)
    other = 1 - published_base
    if rates[other] > rates[published_base] + NUMBERING_MARGIN:
        raise RuntimeError(
            f"Evidence matches {rates[other]:.0%} of sampled pages counting from {other} but only "
            f"{rates[published_base]:.0%} counting from {published_base}; the page numbering "
            "assumed for this dataset is wrong")
    return {f"base{base}": round(rate, 4) for base, rate in rates.items()}


def text_layer(evidence: tuple[tuple[str, int], ...], page_texts: dict[str, list[str]]) -> str:
    """"scanned" when an evidence page has no text layer, which table recognition skips."""
    blank = any(not page_texts[document][page - 1].strip() for document, page in evidence)
    return "scanned" if blank else "text"


def validate(documents: list[PdfDocument], questions: list[PdfQuestion]) -> None:
    pages = {document.id: document.pages for document in documents}
    if len(pages) != len(documents):
        raise ValueError("Document IDs repeat")
    if len({question.id for question in questions}) != len(questions):
        raise ValueError("Question IDs repeat")
    for question in questions:
        if not question.evidence:
            raise ValueError(f"{question.id} has no evidence pages")
        for document, page in question.evidence:
            if document not in pages:
                raise ValueError(f"{question.id} names unknown document {document}")
            if not 1 <= page <= pages[document]:
                raise ValueError(f"{question.id} cites page {page} of {document}, "
                                 f"which has {pages[document]} pages")


def query_groups(question: PdfQuestion) -> dict[str, str]:
    """The group labels group_metrics.py breaks a run down by."""
    groups = {"source": question.source, "type": question.type, "textLayer": question.text_layer}
    if question.table is not None:
        groups["table"] = "yes" if question.table else "no"
    return groups


def write_pdf_dataset(output: Path, documents: list[PdfDocument], questions: list[PdfQuestion],
                      manifest_fields: dict) -> dict:
    """Write a BEIR PDF dataset with a page-scored "test" split and a document-scored "doc" split."""
    validate(documents, questions)
    hashes = {document.id: hashlib.sha256(document.source.read_bytes()).hexdigest()
              for document in documents}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Two PDFs are byte-identical; the app would store them as one document")

    (output / "pdfs").mkdir(parents=True)
    (output / "qrels").mkdir()
    manifest_documents = []
    for document in documents:
        relative = f"pdfs/{document.id}.pdf"
        shutil.copy2(document.source, output / relative)
        manifest_documents.append({"id": document.id, "file": relative, "pages": document.pages,
                                   "sha256": hashes[document.id]})

    write_jsonl(output / "corpus.jsonl", [
        record
        for document in documents
        for record in [{"_id": document.id, "title": document.id, "text": ""}] + [
            {"_id": page_id(document.id, page), "title": document.id, "text": ""}
            for page in range(1, document.pages + 1)
        ]
    ])
    write_jsonl(output / "queries.jsonl",
                [{"_id": question.id, "text": question.text} for question in questions])
    header = "query-id\tcorpus-id\tscore"
    page_rows = [f"{question.id}\t{page_id(document, page)}\t1"
                 for question in questions for document, page in sorted(set(question.evidence))]
    document_rows = [f"{question.id}\t{document}\t1"
                     for question in questions
                     for document in sorted({document for document, _ in question.evidence})]
    (output / "qrels" / "test.tsv").write_text("\n".join([header, *page_rows]) + "\n", encoding="utf-8")
    (output / "qrels" / "doc.tsv").write_text("\n".join([header, *document_rows]) + "\n", encoding="utf-8")
    (output / "query-groups.json").write_text(json.dumps(
        {question.id: query_groups(question) for question in questions}, indent=2) + "\n",
        encoding="utf-8")
    (output / "answers.json").write_text(json.dumps(
        {question.id: question.answer for question in questions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    manifest = {"unit": "page", "splitUnits": {"test": "page", "doc": "document"},
                **manifest_fields, "documents": manifest_documents}
    (output / PDF_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"documents": len(documents), "pages": sum(document.pages for document in documents),
            "queries": len(questions), "testQrels": len(page_rows),
            "scannedEvidenceQueries": sum(question.text_layer == "scanned" for question in questions)}


def prepare_output(output: Path, replace: bool) -> None:
    if output.exists():
        if not replace:
            raise SystemExit(f"{output} exists; pass --replace to rebuild it")
        shutil.rmtree(output)
