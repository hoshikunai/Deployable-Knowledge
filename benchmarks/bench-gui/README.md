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

## Embedding models

The **Embedding model** dropdown on the Run tab picks which model's corpus copies jobs use:
nomic-embed-text v1.5 (the app's default, 768-d), bekko-embedding v1 a25m (384-d), bge-base-en v1.5
(768-d), bge-small-en v1.5 (384-d), granite-embedding 97m multilingual r2 (384-d), granite-embedding
english r2 (768-d), or jina-embeddings v5 text nano retrieval (768-d; CC BY-NC 4.0, so for
benchmarking only, not as the app's default). Each model
needs its own copy of every corpus, because the stored vectors come from the model; corpora
without one show a **Prepare** button. Copies are named by what differs from the default, e.g.
`runtime-scifact-bekko-a25m` or `runtime-scifact-t256o32-bekko-a25m`. The app reads the model from
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

## Terminal commands

```bash
npm run bench -- run scifact arguana --pipeline rrf-ettin --queries 100 --seeds 42,314
npm run bench -- run nfcorpus --pipeline rrf-ettin --rerank-max-tokens 1024 --chunk-tokens 256 --overlap-tokens 32
npm run bench -- prepare nfcorpus --chunk-tokens 256 --overlap-tokens 32
npm run bench -- prepare scifact --embedding-model bekko-embedding-v1-a25m
npm run bench -- run scifact --pipeline rrf --embedding-model bekko-embedding-v1-a25m
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
