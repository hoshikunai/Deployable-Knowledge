# BEIR retrieval harness

This harness evaluates the application's existing `/search` endpoint without
replacing the RAG pipeline. BEIR supplies a corpus, queries, and relevance
judgments. The application supplies BM25, semantic, and hybrid rankings.

BEIR evaluates retrieval only. It does not call Ollama, llama.cpp, or GitHub
Models, and it does not judge generated answers.

## Install the Python environment

From the repository root:

```bash
python3 -m venv benchmarks/beir/.venv
benchmarks/beir/.venv/bin/python -m pip install \
  -r benchmarks/beir/requirements.txt
```

## Download additional corpora

A BEIR corpus is installed by downloading and extracting its dataset archive;
there is no separate Python package per corpus.

List the recommended datasets:

```bash
benchmarks/beir/.venv/bin/python \
  benchmarks/beir/download_datasets.py --list
```

Download several manageable corpora:

```bash
benchmarks/beir/.venv/bin/python \
  benchmarks/beir/download_datasets.py nfcorpus arguana
```

Download the larger FiQA corpus later:

```bash
benchmarks/beir/.venv/bin/python \
  benchmarks/beir/download_datasets.py fiqa
```

The archives and extracted files are placed under
`benchmarks/beir/datasets/`, which is ignored by Git.

Recommended starting datasets:

| Name       | Corpus | Test queries | What it probes                    |
| ---------- | -----: | -----------: | --------------------------------- |
| `scifact`  |     5K |          300 | Scientific claims and evidence    |
| `nfcorpus` |   3.6K |          323 | Biomedical/nutrition questions    |
| `arguana`  |  8.67K |        1,406 | Counterargument matching          |
| `scidocs`  |    25K |        1,000 | Scientific citation relationships |
| `fiqa`     |    57K |          648 | Financial questions and answers   |

These names, splits, and approximate sizes come from the
[official BEIR dataset table](https://github.com/beir-cellar/beir#available-datasets).
Check that table before attempting a very large corpus such as NQ, HotpotQA,
or MSMARCO. Ingestion creates embeddings for every document and may take much
longer than the archive download.

Any public single-directory BEIR dataset can be downloaded by name:

```bash
benchmarks/beir/.venv/bin/python \
  benchmarks/beir/download_datasets.py DATASET_NAME
```

The downloader uses BEIR's official URL convention. A dataset still needs the
standard `corpus.jsonl`, `queries.jsonl`, and `qrels/<split>.tsv` layout.

## Run the automated test pipeline

### One-command prepared-runtime benchmark

For the three corpora already indexed on this machine, use the guarded runner
from the repository root. It builds the application once, validates that each
saved document mapping matches its isolated runtime, and executes one dataset
at a time. It never ingests or downloads documents.

```bash
# Inspect the planned command without building or searching.
benchmarks/beir/.venv/bin/python benchmarks/beir/run_prepared_benchmarks.py --dry-run

# Run the full ArguAna test split (the default dataset).
benchmarks/beir/.venv/bin/python benchmarks/beir/run_prepared_benchmarks.py

# Or run all three prepared corpora sequentially.
benchmarks/beir/.venv/bin/python benchmarks/beir/run_prepared_benchmarks.py \
  scifact nfcorpus arguana

# Evaluate rankings through 100 unique documents, including Recall@100.
benchmarks/beir/.venv/bin/python benchmarks/beir/run_prepared_benchmarks.py \
  scifact nfcorpus --document-depth 100
```

`--document-depth` defaults to 10 and accepts 10–100. It changes the result
cutoff, not the number of test queries. At depth 100 the harness asks the
application for 200 chunks, collapses them to at most 100 unique documents per
method, and records the number actually available in `run-summary.json` under
`rankingCoverage`. Some queries can return fewer than 100 distinct documents;
their Recall@100 is calculated from the available ranking. This remains a
chunk-first application protocol, not Anserini's flat-document indexing
protocol. Deeper searches require much more cross-encoder work and may need
more bounded sessions. Use `--dry-run --document-depth 100` to check the
command first. Resume with the same `--document-depth` value.

Each session is capped at 45 minutes. A session that times out after saving new
query checkpoints is resumed automatically, up to eight sessions by default.
Memory pauses, failed searches, a missing mapping, and timeouts with no new
checkpoints stop the script for inspection. Increase the limit with
`--max-sessions N`; use `--skip-build` only when `build/index.js` is current.

To continue a run stopped after the session limit, use the run name printed in
the output and keep the same timeout and mapping arguments:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/run_prepared_benchmarks.py \
  arguana --resume --run-name RUN_NAME
```

This command uses saved local mappings; a fresh clone does not contain them.
Supply another validated mapping with `--mapping DATASET=PATH`. The runner
refuses to resume a failure that occurred before BEIR created its run config;
start a new run instead. Results are local BEIR retrieval evaluations, not
automatically equivalent to a published leaderboard's indexing or top-K
protocol. No answer generation or reranker-only evaluation is included.

### General suite runner

Build the SvelteKit adapter-node server first:

```bash
npm run build:electron
```

Then run a first cross-dataset suite:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/run_suite.py \
  scifact nfcorpus \
  --split test \
  --query-counts 10,50,100,all \
  --sample-seed 42 \
  --search-depth 10
```

The suite runner does the following for each dataset:

1. Downloads the corpus when necessary.
2. Creates `.cache/beir/runtime-<dataset>-001/`.
3. Uses a separate `app.db` so corpora cannot contaminate one another.
4. Links the existing Transformers.js model cache when available.
5. Starts the TypeScript server on port 4179.
6. Ingests the corpus and runs the requested queries.
7. Stops the server before moving to the next dataset.
8. Writes a combined suite summary under `benchmarks/beir/runs/suites/`.

Some BEIR corpora, including NFCorpus, contain different document IDs with
identical title and text. The application stores one content-deduplicated
document for those IDs. The harness records all aliases and merges their qrels
into the first canonical BEIR ID so identical content is evaluated once rather
than causing an ingestion failure.

Stop any server already using port 4179 before starting a suite. Use `--port`
to select a different port when necessary.

Run one quick smoke test before committing to a full suite:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/run_suite.py \
  nfcorpus \
  --query-counts 10 \
  --search-depth 10
```

### Why the query counts are nested

For `10,50,100,all`, the runner shuffles the query IDs once using the recorded
seed. The first 10 queries are contained in the first 50, which are contained
in the first 100. Each query is searched only once.

This makes it possible to see when the metrics begin to stabilize:

- 10 queries: smoke testing only; scores can move dramatically.
- 50 queries: useful for early directional feedback.
- 100 queries: better for comparing substantial changes.
- All queries: the result to report and preserve as the baseline.

Use the same seed when comparing two retrieval implementations. Changing both
the code and the sampled queries makes the comparison ambiguous.

### Search-depth warning

`--search-depth` is the number of chunks requested before results are collapsed
to unique BEIR documents. Depth 10 is safe on the current hardware but can
produce fewer than ten unique documents. Depth 100 caused the current hybrid
cross-encoder to exhaust memory because it reranked hundreds of passages in one
batch. Do not use a large depth until reranking is batched.

## Run a single dataset against a manually started server

The original workflow remains available:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/run.py \
  --dataset scifact \
  --split test \
  --base-url http://127.0.0.1:4179 \
  --search-depth 10 \
  --query-counts 10,50,100,all \
  --sample-seed 42
```

`--query-limit N` still selects the first N dataset-ordered queries for
backward compatibility. Prefer `--query-counts` for comparisons because it
uses a reproducible sample instead of a potentially biased prefix.

Each completed run writes:

- `metrics.json`: metrics for the largest checkpoint.
- `metrics-by-query-count.json`: metrics at every requested checkpoint.
- `query-selection.json`: exact query IDs, order, seed, and checkpoints.
- `document-rankings.json`: document-level rankings used by BEIR.
- `chunk-rankings.jsonl`: raw application chunk rankings.
- `document-id-mapping.json`: application-to-BEIR ID mapping.
- `run-config.json`: the complete evaluation protocol.

## Use train/dev/test correctly

Use a dataset's `train` or `dev` split while choosing retrieval settings. Run
the `test` split after selecting the design. Repeatedly selecting changes based
on the test score leaks information from the test set into the design.

For example, NFCorpus supports all three splits:

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/run_suite.py \
  nfcorpus --split dev --query-counts 50,100,all
```

ArguAna only supplies a test split, so use it for confirmation rather than
iterative tuning.

## Evaluate generated answers

BEIR qrels say which documents are relevant; they generally do not provide a
complete reference answer. Answer evaluation therefore needs a separate set of
questions and expected facts. Start with
`answer-evaluation-template.csv` and create 25–100 cases representative of the
documents users will actually load.

Score each answer from 0 to 2 on these dimensions:

| Dimension         | 0                        | 1                 | 2                      |
| ----------------- | ------------------------ | ----------------- | ---------------------- |
| Correctness       | Materially wrong         | Mixed/minor error | Fully correct          |
| Faithfulness      | Unsupported claims       | Partly grounded   | Every claim supported  |
| Completeness      | Misses the answer        | Partial answer    | Covers required facts  |
| Citation accuracy | Wrong/missing sources    | Mixed             | Sources support claims |
| Abstention        | Invents when unsupported | Unclear hedging   | Correctly abstains     |

Use three controlled conditions to locate failures:

1. **Normal RAG:** use the documents returned by the retrieval pipeline.
2. **Oracle context:** manually provide the known relevant document. If this
   succeeds while normal RAG fails, retrieval is the likely problem.
3. **No context:** remove document context. This reveals what the model answers
   from prior knowledge rather than the supplied evidence.

Keep the provider, model, prompt, temperature, token limit, retrieval mode, and
RAG top-K fixed while comparing conditions. Save the raw answer and retrieved
document IDs with every score.

Manual grading by a domain-aware reviewer is the most trustworthy starting
point. An LLM judge can accelerate larger evaluations, but blind it to the
system name, randomize answer order, require a short justification, and
manually audit a sample. Never let the same model grade itself without an
independent check.

Retrieval and answer metrics answer different questions:

- BEIR: "Did the system retrieve relevant evidence?"
- Answer evaluation: "Did the model use that evidence correctly?"

A production-ready RAG evaluation needs both.
