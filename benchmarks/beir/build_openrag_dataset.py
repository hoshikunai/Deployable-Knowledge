"""Build a BEIR PDF dataset from Vectara's Open RAG Benchmark (arXiv papers).

Each Open RAG query is labelled with one paper and one logical section. The builder writes two
qrel splits over the same PDFs:

- test: pages of the relevant section, located by aligning the section text with the text
  pdf-parse reads from each page. Queries whose section cannot be located are left out.
- doc: the relevant paper as a whole.

Papers are a seeded sample of the positive (queried) and hard-negative papers; --all takes
every paper. Downloads are cached, so rebuilding a different sample reuses them.

    benchmarks/beir/.venv/bin/python benchmarks/beir/build_openrag_dataset.py
    benchmarks/beir/.venv/bin/python benchmarks/beir/build_openrag_dataset.py \\
        --name openrag-smoke --positives 5 --negatives 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.request
from pathlib import Path

from build_pdf_dataset import REPOSITORY_ROOT, page_id, write_jsonl
from dataset_catalog import PDF_MANIFEST, validate_dataset_name

HARNESS_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = HARNESS_ROOT / "datasets"
SOURCE_URL = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
METADATA_FILES = ("queries.json", "qrels.json", "answers.json", "pdf_urls.json")
USER_AGENT = "beir-harness-pdf-builder/1.0"
ARXIV_DELAY_SECONDS = 3.0
SHINGLE_SIZE = 6
ANCHOR_SHINGLES = 20
MIN_ANCHOR_MATCHES = 3
ALIGNMENT_WARNING_SHARE = 0.8

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

MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
PLACEHOLDER = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HYPHEN_BREAK = re.compile(r"-\s*\n\s*")
WORD = re.compile(r"[a-z]{3,}")


def fetch(url: str, destination: Path, expected_prefix: bytes | None = None) -> bool:
    """Download url to destination unless it is cached. Returns whether it hit the network."""
    if destination.is_file():
        return False
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    if expected_prefix is not None and not body.startswith(expected_prefix):
        raise RuntimeError(f"{url} did not return the expected file type")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(body)
    partial.replace(destination)
    return True


def load_metadata(cache: Path) -> dict[str, dict]:
    metadata = {}
    for name in METADATA_FILES:
        fetch(f"{SOURCE_URL}/{name}", cache / name)
        metadata[name.removesuffix(".json")] = json.loads((cache / name).read_text(encoding="utf-8"))
    return metadata


def select_papers(qrels: dict[str, dict], paper_ids: list[str], positives: int | None,
                  negatives: int | None, seed: int) -> tuple[list[str], list[str]]:
    """A seeded sample of queried and never-queried papers; None takes all of a kind."""
    queried = {label["doc_id"] for label in qrels.values()}
    positive_pool = sorted(queried)
    negative_pool = sorted(set(paper_ids) - queried)
    generator = random.Random(seed)
    for pool, wanted, kind in ((positive_pool, positives, "positive"),
                               (negative_pool, negatives, "negative")):
        if wanted is not None and wanted > len(pool):
            raise ValueError(f"Only {len(pool)} {kind} papers exist; asked for {wanted}")
    chosen_positives = positive_pool if positives is None else generator.sample(positive_pool, positives)
    chosen_negatives = negative_pool if negatives is None else generator.sample(negative_pool, negatives)
    return sorted(chosen_positives), sorted(chosen_negatives)


def download_papers(positives: list[str], negatives: list[str], pdf_urls: dict[str, str],
                    cache: Path) -> None:
    for paper in positives:
        fetch(f"{SOURCE_URL}/corpus/{paper}.json", cache / "corpus" / f"{paper}.json")
    papers = positives + negatives
    for index, paper in enumerate(papers, start=1):
        if fetch(pdf_urls[paper], cache / "pdfs" / f"{paper}.pdf", expected_prefix=b"%PDF-"):
            print(f"Downloaded {index}/{len(papers)} PDFs")
            time.sleep(ARXIV_DELAY_SECONDS)


def read_page_texts(files: list[Path]) -> dict[Path, list[str]]:
    """Per-page text from pdf-parse, the parser the app's PDF ingestion uses."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PAGE_TEXT_SCRIPT, *map(str, files)],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True,
    )
    texts = json.loads(result.stdout)
    return {path: texts[str(path)] for path in files}


def normalize_words(text: str) -> list[str]:
    """Words shared by Mistral OCR Markdown and pdf-parse text; math and markup are dropped."""
    text = unicodedata.normalize("NFKC", text)
    text = PLACEHOLDER.sub(" ", MATH.sub(" ", text))
    text = HYPHEN_BREAK.sub("", text)
    return WORD.findall(text.lower())


def shingles(words: list[str]) -> list[tuple[str, ...]]:
    return [tuple(words[index:index + SHINGLE_SIZE])
            for index in range(len(words) - SHINGLE_SIZE + 1)]


def align_sections(sections: list[str], pages: list[str]) -> list[tuple[int, int] | None]:
    """
    Locate each section's 1-based page range in a PDF.

    Sections are walked in order; a section starts on the first page at or after the previous
    section's start that holds enough of its opening word shingles. It runs to the page where
    the next located section starts (inclusive, since headings usually begin mid-page), or to
    the last page. Sections whose opening is not found are None.
    """
    page_shingles = [set(shingles(normalize_words(text))) for text in pages]
    starts: list[int | None] = []
    floor = 0
    for text in sections:
        anchor = shingles(normalize_words(text))[:ANCHOR_SHINGLES]
        start = None
        if len(anchor) >= MIN_ANCHOR_MATCHES:
            start = next((page for page in range(floor, len(pages))
                          if sum(item in page_shingles[page] for item in anchor) >= MIN_ANCHOR_MATCHES),
                         None)
        if start is not None:
            floor = start
        starts.append(start)

    ranges: list[tuple[int, int] | None] = []
    for index, start in enumerate(starts):
        if start is None:
            ranges.append(None)
            continue
        following = next((later for later in starts[index + 1:] if later is not None), None)
        end = len(pages) - 1 if following is None else following
        ranges.append((start + 1, end + 1))
    return ranges


def build(output: Path, cache: Path, positives: int | None, negatives: int | None,
          seed: int) -> dict:
    metadata = load_metadata(cache)
    qrels, queries = metadata["qrels"], metadata["queries"]
    positive_papers, negative_papers = select_papers(
        qrels, list(metadata["pdf_urls"]), positives, negatives, seed)
    download_papers(positive_papers, negative_papers, metadata["pdf_urls"], cache)
    papers = positive_papers + negative_papers
    sources = {paper: cache / "pdfs" / f"{paper}.pdf" for paper in papers}
    texts = read_page_texts(list(sources.values()))
    page_counts = {paper: len(texts[path]) for paper, path in sources.items()}
    if empty := [paper for paper, count in page_counts.items() if count == 0]:
        raise RuntimeError(f"pdf-parse read no pages from {', '.join(empty)}")

    section_ranges = {}
    for paper in positive_papers:
        document = json.loads((cache / "corpus" / f"{paper}.json").read_text(encoding="utf-8"))
        section_ranges[paper] = align_sections(
            [section["text"] for section in document["sections"]], texts[sources[paper]])

    selected = sorted(query for query, label in qrels.items() if label["doc_id"] in section_ranges)
    query_ids = {query: f"{queries[query]['source']}-{query}" for query in selected}
    page_rows, document_rows, unaligned = [], [], []
    for query in selected:
        paper, section = qrels[query]["doc_id"], qrels[query]["section_id"]
        document_rows.append(f"{query_ids[query]}\t{paper}\t1")
        ranges = section_ranges[paper]
        if section >= len(ranges) or ranges[section] is None:
            unaligned.append(query_ids[query])
            continue
        first, last = ranges[section]
        page_rows += [f"{query_ids[query]}\t{page_id(paper, page)}\t1"
                      for page in range(first, last + 1)]

    (output / "pdfs").mkdir(parents=True)
    (output / "qrels").mkdir()
    documents = []
    for paper in papers:
        relative = f"pdfs/{paper}.pdf"
        shutil.copy2(sources[paper], output / relative)
        documents.append({
            "id": paper, "file": relative, "pages": page_counts[paper],
            "sha256": hashlib.sha256(sources[paper].read_bytes()).hexdigest(),
            "role": "positive" if paper in section_ranges else "negative",
        })

    write_jsonl(output / "corpus.jsonl", [
        record
        for paper in papers
        for record in [{"_id": paper, "title": paper, "text": ""}] + [
            {"_id": page_id(paper, page), "title": paper, "text": ""}
            for page in range(1, page_counts[paper] + 1)
        ]
    ])
    write_jsonl(output / "queries.jsonl",
                [{"_id": query_ids[query], "text": queries[query]["query"]} for query in selected])
    header = "query-id\tcorpus-id\tscore"
    (output / "qrels" / "test.tsv").write_text("\n".join([header, *page_rows]) + "\n", encoding="utf-8")
    (output / "qrels" / "doc.tsv").write_text("\n".join([header, *document_rows]) + "\n", encoding="utf-8")
    (output / "query-groups.json").write_text(json.dumps({
        query_ids[query]: {"source": queries[query]["source"], "type": queries[query]["type"]}
        for query in selected
    }, indent=2) + "\n", encoding="utf-8")
    (output / "answers.json").write_text(json.dumps({
        query_ids[query]: metadata["answers"][query] for query in selected
    }, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "unit": "page",
        "splitUnits": {"test": "page", "doc": "document"},
        "source": "vectara/open_ragbench pdf/arxiv (CC-BY-NC-4.0)",
        "seed": seed,
        "documents": documents,
        "alignment": {"aligned": len(selected) - len(unaligned), "unaligned": len(unaligned),
                      "unalignedQueryIds": unaligned},
    }
    (output / PDF_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"positivePapers": len(positive_papers), "negativePapers": len(negative_papers),
            "pages": sum(page_counts.values()), "docQueries": len(selected),
            "testQueries": len(selected) - len(unaligned), "testQrels": len(page_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="openrag-arxiv", help="Dataset folder name.")
    parser.add_argument("--positives", type=int, default=100, help="Queried papers to sample.")
    parser.add_argument("--negatives", type=int, default=150, help="Hard-negative papers to sample.")
    parser.add_argument("--all", action="store_true", help="Take every paper (1,000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-cache", type=Path, default=DATASETS_ROOT / ".openrag-source",
                        help="Where downloaded metadata, section files and PDFs are kept.")
    parser.add_argument("--replace", action="store_true", help="Rebuild an existing dataset folder.")
    arguments = parser.parse_args()
    output = DATASETS_ROOT / validate_dataset_name(arguments.name)
    if output.exists():
        if not arguments.replace:
            raise SystemExit(f"{output} exists; pass --replace to rebuild it")
        shutil.rmtree(output)
    positives = None if arguments.all else arguments.positives
    negatives = None if arguments.all else arguments.negatives
    summary = build(output, arguments.source_cache.expanduser().resolve(), positives, negatives,
                    arguments.seed)
    print(f"Built {output}: {json.dumps(summary)}")
    share = summary["testQueries"] / summary["docQueries"] if summary["docQueries"] else 0.0
    if share < ALIGNMENT_WARNING_SHARE:
        print(f"Warning: only {share:.0%} of queries were aligned to pages; "
              "check the test split before trusting it")


if __name__ == "__main__":
    main()
