# Why TableFormer improves semantic retrieval on pdf-eval

This compares PDF ingestion with **Settings → Documents → Table recognition** (Docling's Heron
layout model plus TableFormer) against the PDF chunking on upstream `main`
(`Deployable-Knowledge/Deployable-Knowledge`, `d1f5ce11`). Everything else is held fixed.

## Short answer

Table recognition changes no existing chunk. It **adds** one TABLE chunk per detected table, where
every row is a line and every value is labelled with its column header (`Header: value; …`).
On `pdf-eval` that is 420 extra chunks on 201 pages. Semantic search improves because, for table
questions, these chunks match the question better than the page text does. The whole gain comes
from the 16 table questions; the other 65 questions are unchanged.

## What is held fixed

Verified before the runs below:

- **The baseline chunker is upstream's.** Extraction and chunking of all 8 eval PDFs on this
  branch, with table recognition off, produce chunks identical to upstream `main`'s (2,825 chunks,
  compared by type, page, order and text).
- **Retrieval is upstream's.** BM25, semantic search, the vector and full-text indexes, union
  fusion, Ettin 32M reranking at 512 tokens, and the final top-K cut are the same code and
  settings. The union pipeline has no fusion weights.
- **The embedding model** is nomic-embed-text v1.5 for both.
- **OCR** is the same: both copies have the same 43 IMAGE chunks.

## What changes in the PDF pipeline

With table recognition on, `extractText` in `src/lib/server/rag/chunk/text-extract.ts` adds one
step per page that has a text layer (`collectRecognizedTables`, `tables/table-extract.ts`):

1. **Find tables.** The page is rendered and run through Docling's Heron layout model (ONNX).
   Regions labelled as tables are kept; a table mostly inside a larger one is dropped.
2. **Read their structure.** Each table region is rendered at Docling's table scale and run
   through a TypeScript port of TableFormer's inference (ONNX exports of the `accurate` model).
   It predicts rows, columns, row and column spans, header cells, and a box for each cell.
3. **Fill the cells.** The page's own PDF text is placed into the cells by overlap
   (`tables/table-cells.ts`), so no OCR is involved and the words are exact.
4. **Write the table** (`serializeTable`): one line per body row. When the table has column
   headers, every value is written as `Header: value`, joined with `; `. Values that span rows are
   repeated on each row they cover.
5. **Chunk it at row boundaries.** The chunk is marked `rowPerLine`, so `chunker.ts` packs whole
   rows into chunks of up to 1,200 characters instead of cutting mid-row
   (`splitTableRows`).

On those pages, pdf-parse's own table detection is skipped. pdf-parse finds no tables in these
PDFs either way, so the baseline has 0 TABLE chunks. **The page text is indexed unchanged**, so
every TEXT chunk is the same in both copies.

| Copy                                      | TEXT  | IMAGE | TABLE | Total |
| ----------------------------------------- | ----- | ----- | ----- | ----- |
| Upstream chunking (table recognition off) | 2,782 | 43    | 0     | 2,825 |
| Table recognition on                      | 2,782 | 43    | 420   | 3,245 |

TABLE chunks are a little shorter than TEXT chunks (median 782 characters against 1,084).

## Results

These are two complete runs on all 81 queries: the app default pipeline, scored by page at depth 10.

- Upstream chunking: `pdf-eval-default-p40-o20-all-20261116-180551-878808`
- Table recognition: `pdf-eval-default-p40-o20-docling-all-20261116-180013-146819`

| Method   | nDCG@10             | Recall@10           | MRR@10              |
| -------- | ------------------- | ------------------- | ------------------- |
| BM25     | 0.7165 → 0.7240     | 0.8529 → 0.8529     | 0.7077 → 0.7176     |
| Semantic | 0.6463 → **0.6714** | 0.8056 → **0.8272** | 0.6130 → **0.6397** |
| Hybrid   | 0.7871 → 0.7971     | 0.9053 → 0.9115     | 0.7810 → 0.7971     |

The table below gives the change in nDCG@10, split by question type. The p-values come from a
paired sign-flip test on per-query nDCG@10.

| Method   | All 81            | Table questions (X, 16) | Other questions (F and T, 65) |
| -------- | ----------------- | ----------------------- | ----------------------------- |
| Semantic | +0.025 (p = 0.11) | **+0.135 (p = 0.09)**   | −0.002                        |
| Hybrid   | +0.010 (p = 0.51) | +0.066 (p = 0.34)       | −0.004                        |
| BM25     | +0.008 (p = 0.75) | +0.038 (p = 0.75)       | 0.000                         |

**Semantic, first relevant page:** 5 queries improved, 1 got worse. In every case the relevant hit
was a TABLE chunk.

| Query | Before        | After |
| ----- | ------------- | ----- |
| X1    | 4             | 1     |
| X3    | 3             | 1     |
| X8    | not in top 10 | 2     |
| X10   | not in top 10 | 2     |
| X14   | 4             | 2     |
| X4    | 1             | 2     |

## Why the table chunks embed better

The three examples below come from the rebuilt corpus copies. In each, the TEXT chunk on the
same page already holds the answer, but as a flattened word stream.

**Values carry their meaning.** pdf.js reads a table as the header row followed by the cell
values in reading order, so a value ends up far from the header that names it. X1 asks: _"What
pitch change gives a 500 feet per nautical mile descent?"_

- The page text reads `Descent Gradient Pitch Change Power Setting Configuration 200 to 250
feet/nm 2 to 2.5 degrees … 500 feet/nm 5 degrees 80% RPM …`.
- The TABLE chunk reads `Descent: 500 feet/nm; Pitch Change: 5 degrees; Power Setting: 80% RPM`.
  The question's terms sit right beside the answer.

**Rows don't blend.** A long table spreads across sentence windows that mix neighbouring rows.
X8 asks: _"What makes a flight lead's pitchout unsatisfactory on a checkride?"_

- The page's TEXT chunk runs items 66, 67 and 68 together (fluid maneuvering, tactical, pitchout).
- The TABLE chunk keeps item 68 apart: `Area: Pitchout (Lead); … U: Aircraft control resulted in
a wingman not able to establish spacing.`

**Tables separate from surrounding prose.** A table and the unrelated paragraph after it often
share one sentence window, which dilutes the embedding. On X1's page, the table and section 7.15
(VORTAC penetration) share the first TEXT chunk. X10 asks: _"How does the citation open for a
Distinguished Flying Cross awarded for achievement?"_ The TEXT chunks on its page are mostly
about order closes. The TABLE chunk is only the decoration matrix, with each cell labelled
`DECORATION`, `HEROISM`, `ACHIEVEMENT` or `SERVICE`.

The same reasons explain why the effect is mostly semantic. BM25 matches terms, and those are
already present in the page text. The new chunks change term proximity and chunk length, but not
term presence, so BM25 moves little: Recall@10 is identical, and only X1, X3 and X4 change rank.

## Why hybrid gains less than semantic

Hybrid reranks the union of the top 40 chunks from each retriever with Ettin 32M.

- **BM25 already finds most of these pages.** Hybrid starts higher (0.787), so there is less room.
- **The extra chunks compete for the same slots.** On the 65 non-table questions, hybrid moved
  slightly down (−0.004). Six queries got worse (F5, T10, T18, X7, X9, X16), against seven that
  improved; T10 dropped out of the top 10. The page text is unchanged, so these moves come from
  table chunks taking candidate and top-10 slots.

## Limits

- **Small sample.** There are only 16 table questions, and the semantic gain on them is
  p = 0.09. The hybrid change is not significant. One question set and one corpus are not enough
  to generalize from.
- **The questions are synthetic.** Claude wrote both question sets by reading single pages (see
  `DOCLING_PORT_EVALUATION.md`).
- **Ingestion is slower.** With table recognition on, ingesting the 8 PDFs took about 18 minutes,
  against about 8 without it.
- **Earlier pdf-eval runs used other chunking.** Runs before 2026-11-16 17:00 UTC used a chunker
  that also stripped running headers, joined sentences across pages and dropped duplicate tables.
  Those copies are kept as `runtime-pdf-eval-*-step1-cleanup`, but their numbers aren't
  comparable to this baseline: there, hybrid went 0.790 → 0.823.
