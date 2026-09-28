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
(`corpus`) and in the run name (`-t256o32-`). A prepare job that stops partway leaves its copy
marked "preparing"; preparing again rebuilds it from scratch. Each copy keeps its document
mapping beside its database, so deleting runs can't break a corpus.

## Terminal commands

```bash
npm run bench -- run scifact arguana --pipeline rrf-ettin --queries 100 --seeds 42,314
npm run bench -- run nfcorpus --pipeline rrf-ettin --rerank-max-tokens 1024 --chunk-tokens 256 --overlap-tokens 32
npm run bench -- prepare nfcorpus --chunk-tokens 256 --overlap-tokens 32
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
