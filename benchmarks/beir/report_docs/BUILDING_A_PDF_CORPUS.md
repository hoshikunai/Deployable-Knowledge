# Building a PDF corpus for the BEIR bench

Standard BEIR corpora are plain text (`corpus.jsonl`: an ID, a title and a block of text per
document). They have no PDFs, pages, layout or tables. Docling, TableFormer and the PDF
cleanups only run on real PDF pages, so testing them needs a PDF corpus. This guide explains how
`pdf-eval` was made and how to build another one.

## How `pdf-eval` was made

The source files are in `~/Projects/handoffs/docling/eval-assets/`, from the earlier Docling
port evaluation (`docs/DOCLING_PORT_EVALUATION.md`).

**Documents.** 8 Air Force PDFs (1,071 pages) that were already in the app's `documents/`
folder. `benchmarks/beir/build_pdf_dataset.py` maps each short name to its file there, e.g.
`"afi36-2406": "c43e4d361db42702.pdf"`.

**Questions and correct answers.** Two JSON files, `questions.json` and
`table-questions.json`. Each question looks like this:

```json
{
	"id": "F1",
	"question": "How many days after the close-out date is an evaluation due to the MPF?",
	"document": "afi36-2406",
	"first": 12,
	"last": 12
}
```

**The correct answer is a location, not a text answer**: one document and a page range
(`first` to `last`). The answer wording is never stored or checked.

**Who wrote them.** Claude wrote all three sets during the earlier evaluation:

| Set       | Questions | How it was written                                                                              |
| --------- | --------- | ----------------------------------------------------------------------------------------------- |
| Fact (F)  | 42        | From single pages. Each answer is one detail on one page.                                       |
| Topic (T) | 23        | Written afterwards. Page ranges come from the PDF bookmarks, so any page of the section counts. |
| Table (X) | 16        | Each answer is a table cell on one page.                                                        |

None of the questions came from real users, and nobody independently checked the page ranges.
That limits how far the `pdf-eval` results generalize.

## How ranking and scoring work

1. **Build.** The builder writes BEIR-format files in which every page is a "document":

   | File                | Contents                                                                                         |
   | ------------------- | ------------------------------------------------------------------------------------------------ |
   | `corpus.jsonl`      | One entry per page, e.g. `afi36-2406#p12`. The text is empty because the app reads the real PDF. |
   | `queries.jsonl`     | The question texts.                                                                              |
   | `qrels/test.tsv`    | The answer key: every page from `first` to `last` is relevant (111 rows for 81 questions).       |
   | `pdf-manifest.json` | Documents, page counts and SHA-256 hashes.                                                       |
   | `pdfs/`             | Copies of the PDFs.                                                                              |

2. **Prepare.** The harness uploads each PDF through the app's normal upload endpoint
   (`POST /documents`), so the selected document parsing applies. It checks the app's page
   count against the manifest.
3. **Run.** Each question is searched. Each returned chunk becomes `<document>#p<page>`,
   keeping the first time each page appears, which gives a ranked list of pages.
4. **Score.** The page list is compared with the qrels to give nDCG@10, MRR@10, MAP@10 and
   Recall@10. **A hit is any chunk from a relevant page**, even if that chunk doesn't contain
   the answer. This makes the score independent of how each pipeline cuts its chunks, at the
   cost of being coarse.

## Building your own corpus

### 1. Choose documents for what you want to test

Include at least a few documents of each kind you care about, so one odd document doesn't
dominate:

- scanned PDFs, where the whole page goes through OCR;
- multi-column layouts;
- pages heavy with tables or forms;
- slides exported to PDF;
- documents without running headers;
- several publishers.

### 2. Write questions

Each question has exactly one document and one page range.

- **Use physical page numbers, not printed ones.** Count from the first page of the file (what
  a PDF viewer shows in its page box), not the number printed on the page. These often differ,
  and a wrong page turns a correct answer into a miss.
- **List every page where the answer appears.** If the same fact is also on page 80, page 80
  counts as a miss. The format supports only one continuous range in one document, so either
  write questions with a single location or note the limitation.
- **Don't copy the page's wording.** Paraphrase as a user would ask. Questions built from the
  page's own words favor BM25.
- **Keep each kind of question as its own set** (facts, topics, tables, OCR-only content) and
  score the sets separately. An overall number depends on an arbitrary mix.
- **Have real users write them, if you can.** That is the only way to learn which question
  styles matter.
- **Plan for enough questions.** As a rough guide, which is an estimate rather than a
  calculation, aim for at least 40–50 per kind you want to draw conclusions about. With 16, a
  few questions swing the p-values.

### 3. Check the answer key

Open each PDF at each range and confirm the answer is there, and not obviously somewhere else.
For scanned documents, also check that the answer is readable text: if Tesseract can't read it,
no pipeline can find it, and the question measures OCR rather than retrieval.

### 4. Save the questions

Use the JSON format above (`id`, `question`, `document`, `first`, `last`). IDs must be unique
across files; the builder refuses repeats and page ranges outside a document.

### 5. Build the dataset

**`build_pdf_dataset.py` is currently hard-coded to `pdf-eval`.**

- The dataset name, the 8-file `CORPUS` map, the source folder (`documents/`) and the question
  file names are fixed in the script.
- A new corpus also needs a `DatasetInfo` entry in `DATASETS` in
  `benchmarks/beir/dataset_catalog.py`, or the GUI won't list it.

Until the script takes a name, a PDF folder and question files as options, build a new corpus
by copying the script and changing those constants. Then add the catalog entry. Its output goes
to `benchmarks/beir/datasets/<name>/`.

### 6. Prepare and run

Prepare one copy per parsing mode (Current chunking, Docling + TableFormer, Docling +
TableFormer + PDF cleanups), run each, and compare them in the Scores tab:

```bash
npm run bench -- prepare <name>
npm run bench -- prepare <name> --document-parsing docling
npm run bench -- prepare <name> --document-parsing docling-cleanup
npm run bench -- run <name> --document-parsing docling
```

Docling ingestion needs a lot of memory (about 5 GB on `pdf-eval`). Close other heavy programs
first, or the harness's memory safeguard pauses the job.

### 7. Interpret honestly

- Look at each question set separately, with the paired p-values.
- Read the per-question rank changes before trusting an average.
- Repeat on a second corpus before drawing conclusions.

## The Open RAG Benchmark corpus (`openrag-arxiv`)

`build_openrag_dataset.py` builds a second, larger PDF corpus from Vectara's
[Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench): 1,000 arXiv papers
(396 with queries, the rest hard negatives that no query points to) and 3,045 queries. The
license is CC-BY-NC-4.0, and each arXiv paper keeps its own license, so the built folder and the
download cache are ignored by Git and must not be committed.

```bash
# Default: a seeded sample of 100 queried papers and 150 negatives.
benchmarks/beir/.venv/bin/python benchmarks/beir/build_openrag_dataset.py

# A quick smoke corpus, or every paper.
benchmarks/beir/.venv/bin/python benchmarks/beir/build_openrag_dataset.py \
  --name openrag-smoke --positives 5 --negatives 5
benchmarks/beir/.venv/bin/python benchmarks/beir/build_openrag_dataset.py --name openrag-all --all
```

Metadata, section files and PDFs are cached in `benchmarks/beir/datasets/.openrag-source/`.
arXiv is asked for one PDF every three seconds, so the first default build takes about 15 minutes;
later builds reuse the cache. A name other than `openrag-arxiv` also needs a `DatasetInfo` entry
in `dataset_catalog.py` before the GUI lists it. The command line works without one.

### Two splits over the same PDFs

Each Open RAG query is labelled with one paper and one logical section (such as "3. Method"), not
a page. The builder writes two qrel splits, and `pdf-manifest.json` (`splitUnits`) tells the
harness how to rank each one:

| Split  | Ranking unit | A hit is                                      |
| ------ | ------------ | --------------------------------------------- |
| `test` | page         | any chunk from a page of the relevant section |
| `doc`  | document     | any chunk from the relevant paper             |

Both splits use the same prepared copy, so prepare once and run with `--split test` and
`--split doc`.

### How sections become page ranges

The section text comes from Mistral OCR (Markdown with LaTeX); the app reads pages with
pdf-parse. The builder drops math, image and table placeholders and short words from both, then
looks for each section's opening six-word sequences in the pages, in order. A section runs from
the page where it starts to the page where the next section starts, inclusive. A section whose
opening isn't found is left out of `test` (its queries stay in `doc`), and the builder warns if
fewer than 80% of queries align. `pdf-manifest.json` lists the unaligned query IDs.

On a 25-paper check, 97% of sections and every query aligned. For 594 of 597 sections, at least
90% of the section's text that pdf-parse reads anywhere in the paper fell inside its page range.
Because the range includes the next section's first page, the `test` split is slightly lenient.

### Query groups

Query IDs start with their source: `text-`, `text-image-`, `text-table-` or `text-table-image-`.
`query-groups.json` also records the type (abstractive or extractive). Score a finished run per
group with:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/group_metrics.py \
  benchmarks/beir/runs/RUN_FOLDER --by source
```

`text-table` queries are the ones most likely to show a TableFormer difference.

### Caveats

- The queries were generated by gpt-4o-mini from the OCR text and filtered with embedding models.
  They may favour semantic search, and they aren't real users' questions.
- Hard negatives were mined to be unrelated to every query, so they make the corpus bigger more
  than they make it harder.
- `answers.json` holds the reference answers for later answer evaluation. The BEIR run ignores it.
