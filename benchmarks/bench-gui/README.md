# BEIR bench

A GUI and short command for running the BEIR harness against prepared corpora, with a
switchable hybrid pipeline. It replaces hand-edited heredocs like
`benchmarks/rag-evaluation/run-command.txt`.

## Start the GUI

```bash
npm run bench
```

Opens `http://127.0.0.1:8765`. Pick corpora, a hybrid pipeline, and all queries or a seeded
sample, then Start. Jobs run one at a time. A session that hits its timeout resumes
automatically; a memory pause, cancel, or closed GUI leaves the run resumable from the Runs tab.
Depths, RRF k, timeouts, ports, and the memory ceiling are under Settings (gear icon).

The Scores tab shows metrics, a paired comparison against the first selected run, and
chunk-score calibration tables. **Copy JSON** copies the raw data; the same JSON is printed in the
terminal.

## Chunk sizes and reranker length

Settings has two token settings:

- **Reranker max tokens** (search time, default 512): how many query + passage tokens the
  reranker reads before truncating. Ettin allows up to 7,999; MS MARCO up to 512. Switching needs
  no rebuild (`RAG_RERANK_MAX_TOKENS`).
- **Chunk size and overlap in tokens** (ingestion time): 0 keeps the app's default
  1,200-character chunks. Any other size needs its own copy of each corpus, because every
  document has to be re-chunked and re-embedded. The Corpus chunking section lists which copies
  exist; **Save and prepare** queues a job that ingests the whole corpus into
  `.cache/beir/runtime-<dataset>-t<size>o<overlap>/`. Overlap carries back whole sentences up to
  that many tokens. Tokens are counted with the embedding model's tokenizer
  (`RAG_CHUNK_MAX_TOKENS`, `RAG_CHUNK_OVERLAP_TOKENS`).

Jobs search the copy that matches the saved chunk setting; runs record it in `run-config.json`
(`corpus`) and in the run name (`-t256o32-`). PDF preparation releases the embedding model
before extraction. Each page layout or table crop runs inference in a fresh process with just
the models it needs. The process exits before the next render. This trades model reload time
for lower peak memory. Resume an interrupted PDF job or prepare the same variant again to reuse
its committed PDFs, including those saved by older harness versions. The worker verifies the PDF
contents and page counts before continuing. Interrupted text copies are rebuilt from scratch.
Each prepared copy keeps its document mapping beside its database.

## Managing corpus copies

The **Corpora** tab lists every copy under `.cache/beir/`, whatever the current settings. It
shows each copy's variant (chunking, embedding model, parsing), state, document and chunk counts,
size, and how many runs searched it.

- **Delete** removes the copy's folder. The links to the shared model caches (`.cache/tables`,
  `.cache/tessdata`) are removed, not the models.
- **Regenerate** deletes the copy and queues a prepare job that rebuilds it with the copy's own
  chunking, embedding model and parsing, not the Run tab's. The job shows on the Run tab.

Both work on default `-001` copies too, with a warning that earlier results were measured on
them. Finished runs keep their scores. Unfinished runs that searched the copy can't be resumed
afterwards, and the confirmation lists them. Both are refused while a running or queued job uses
the copy. Copies whose folder name matches no setting, such as a renamed backup, can only be
deleted.

## Embedding models

The **Embedding model** dropdown on the Run tab picks which model's corpus copies jobs use:
nomic-embed-text v1.5 (the app's llama.cpp default, 768-d) or bge-small-en v1.5 (384-d, a
benchmark-only comparison run through Transformers.js). Each model
needs its own copy of every corpus, because the stored vectors come from the model; corpora
without one show a **Prepare** button. Copies are named by what differs from the default, e.g.
`runtime-scifact-bge-small` or `runtime-scifact-t256o32-bge-small`. The app reads the model from
`RAG_EMBEDDING_MODEL` at startup, and each run confirms the server's model matches its copy.
With token chunking, tokens are counted with the selected model's tokenizer.

To add a model:

1. Create `src/lib/server/rag/embedding-models/<model>.ts`. Models the Transformers.js
   feature-extraction pipeline can run are one `PipelineEmbeddingModel` config (repository, dtype,
   dimensions, max tokens, pooling, query/document prefixes); others implement the
   `EmbeddingModel` interface.
2. List it in `embedding-models/registry.ts`.
3. Add it to `EMBEDDING_MODELS` in `benchmarks/bench-gui/corpora.py` with a short tag for copy
   names and its max tokens.
4. Check its vectors against the official implementation, and add it to `REFERENCES` in
   `benchmarks/embedding-parity/compare.py` first:

   ```bash
   RAG_EMBEDDING_MODEL=<id> node_modules/.bin/tsx benchmarks/embedding-parity/embed.mts app.json
   benchmarks/beir/.venv/bin/python benchmarks/embedding-parity/compare.py app.json
   ```

   It passes when the vectors match (cosine ≥ 0.99 on average) and every test query ranks the
   same top 5. Quantized (8-bit) weights often fail; full precision should match exactly.

5. Rebuild (`npm run build:electron`), then prepare its corpus copies.

## Embedding device

The **Embedding device** choice on the Run tab sets where llama.cpp embeds chunks and queries
(nomic only; bge-small always runs on the CPU):

- **Auto** (the default): the app's own choice, Vulkan when llama.cpp finds a device, else the
  CPU. Existing copies were all made this way.
- **CPU**: llama.cpp on the CPU only (copies named `runtime-<dataset>-cpu`).
- **Vulkan GPU**: llama.cpp on a Vulkan GPU (copies named `runtime-<dataset>-vulkan`). The prepare
  stops before ingesting anything if the app server sees no Vulkan device.

A prepare stores the device in the copy's own settings, so later runs on that copy embed their
queries on the same device. Each run checks the copy's device and the backend that actually
loaded, and records non-default devices in `run-config.json` (`embeddingDevice`).

**Vulkan device** under Settings → Sessions and system limits the app server to one GPU by
index (`GGML_VK_VISIBLE_DEVICES`); `-1` lets it see every device. A prepare refuses an index the
server does not list. Under WSL2 there is currently no Vulkan device: llama.cpp's Vulkan backend
loads but finds nothing, so Auto runs on the CPU and Vulkan GPU fails.

## Document parsing (PDF corpora)

The **Document parsing** choice on the Run tab sets how PDFs are ingested:

- **Current chunking** (`standard`, the default): the app's normal PDF path, with pdf-parse text
  and tables and no table recognition.
- **Docling + TableFormer** (`docling`): the app's table recognition switch
  (`PATCH /table-recognition`) is turned on before ingestion. The Heron layout model finds tables,
  and TableFormer rebuilds their cells.
- **Docling + TableFormer + PDF cleanups** (`docling-cleanup`): Docling + TableFormer, with the
  server started with `RAG_PDF_PAGE_CLEANUP=true`. This benchmark-only flag strips running
  headers and footers (page-edge lines that repeat on at least 30% of pages), joins sentences
  split across pages, drops pdf-parse tables already in the page text, and splits long OCR text.
  It is off in the app because the header strip can delete real text; see
  `benchmarks/beir/report_docs/PDF_CHUNKER_CLEANUPS.md`. Runs check that the server reports the
  same setting (`pdfPageCleanup` in search responses), so rebuild the app after pulling.

OCR (Tesseract on embedded images) is the same in both modes. Each mode has its own corpus copy,
e.g. `runtime-pdf-eval-001`, `runtime-pdf-eval-docling` and
`runtime-pdf-eval-docling-cleanup`. Runs name the mode (`-docling-`, `-docling-cleanup-`) and
record it in `run-config.json` (`documentParsing`).

Only PDF corpora have documents to parse. With either Docling mode selected, text BEIR corpora such as scifact
show as unavailable; they always use the current chunking. The models load from `.cache/tables`
and `.cache/tessdata`, which are linked into every runtime. Download the table models from the
app's Settings first.

### The pdf-eval corpus

This corpus holds the app's 8 eval PDFs (Air Force publications, 1,071 pages) and 81 questions:
42 single-page fact questions (F), 23 topic questions whose answer spans several pages (T), and
16 table questions (X). Results are scored **by page**: a ranked unit is
`<document>#p<page>`, and every page in a question's range counts as relevant. The harness uploads
each PDF through `POST /documents` (multipart), as the app's file picker does, and checks the page
count against the manifest. Build the corpus once from the eval assets:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/build_pdf_dataset.py \
    --eval-assets ~/Projects/handoffs/docling/eval-assets
npm run bench -- prepare pdf-eval
npm run bench -- prepare pdf-eval --document-parsing docling
npm run bench -- run pdf-eval --document-parsing docling
```

## Terminal commands

```bash
npm run bench -- run scifact arguana --pipeline rrf-ettin --queries 100 --seeds 42,314
npm run bench -- run nfcorpus --pipeline rrf-ettin --rerank-max-tokens 1024 --chunk-tokens 256 --overlap-tokens 32
npm run bench -- prepare nfcorpus --chunk-tokens 256 --overlap-tokens 32
npm run bench -- prepare scifact --embedding-model bge-small-en-v1.5
npm run bench -- prepare scifact --embedding-device vulkan --vulkan-device 1
npm run bench -- run scifact --pipeline rrf --embedding-model bge-small-en-v1.5
npm run bench -- resume <run-id>
npm run bench -- list
npm run bench -- scores <run-id> [<run-id> ...] [--markdown]
npm run bench -- delete <run-id>
```

Pipelines: `rrf` (RRF only), `rrf-ettin` (RRF + Ettin 32M), `rrf-msmarco` (RRF + MS MARCO MiniLM).

## How pipelines switch without a rebuild

The app server reads its hybrid pipeline at startup from `RAG_HYBRID_RERANKER`,
`RAG_RETRIEVAL_MULTIPLIER`, `RAG_RERANK_MULTIPLIER`, and `RAG_RRF_K`
(`src/lib/server/rag/search/hybrid-pipeline.ts`). Unset, the app runs RRF only. Each run records
the pipeline in `run-config.json` (`hybridPipeline`) and verifies that the server reports the same
pipeline before searching, so a stale build fails fast. Rebuild with the GUI's **Rebuild app**
button or `npm run build:electron` after changing `src/`.

Run names are generated from the configuration, e.g.
`rrf-ettin-p100-s40-o20-100q-seed42-<timestamp>`: prefetch 100 per retriever, reranker
shortlist 40, 20 output chunks, 100 queries sampled with seed 42.

State lives in `.cache/bench-gui/` (settings and job records); run artifacts stay in
`benchmarks/beir/runs/`.

## Tests

```bash
benchmarks/beir/.venv/bin/python -m unittest discover -s benchmarks/bench-gui
```
