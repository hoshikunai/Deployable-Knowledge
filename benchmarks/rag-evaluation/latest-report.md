# RAG evaluation status

Generated: 2026-09-15T17:54:02.081064+00:00

## Completed

- Existing SciFact provenance inventory recorded with SHA-256 artifact hashes.
- Application RRF utility integrated and produced a 300-query SciFact ranking artifact from the complete saved BM25 and semantic run.
- RRF metrics evaluated with the same `evaluate_method` and canonical SciFact test qrels: nDCG@10 0.72872; bootstrap 95% CI 0.68613–0.76670.
- RRF deltas versus existing full SciFact metrics: +0.05108 BM25, +0.01762 semantic, +0.02850 current hybrid.
- BEIR focused harness: 31 tests passed.
- RRF and HAKARI protocol tests: 4 tests passed.
- `npm run lint`, `npm run check`, and `git diff --check` passed.

## Gates

- Memory safeguards implemented after the WSL OOM: shared fcntl lease, process-group cleanup, verified tree RSS utilities, and conservative preflight gate. No benchmark was resumed.

- HAKARI 0.1.0 installed from commit `e2feac3614a738b17dae5cdb6066d152e4fe3f1c` in Python 3.12.13; custom `predict(pairs)` loader and real loopback scorer fixture passed.
- Official HAKARI NanoBEIR pilot completed for NanoSciFact, NanoNFCorpus, NanoFiQA2018, and NanoArguAna; macro nDCG@10 0.497006.
- HAKARI public DuckDB downloaded and filtered comparison produced for the exact NanoBEIR-en revision and reranking_hybrid profile.
- Full NFCorpus depth-10 execution was attempted with verified runtime reuse; it was interrupted after 18 checkpointed queries, with resume state preserved. No full-corpus score is reported. Exact dry-run/resumable plan is in `full-beir-plan.json`.
- End-to-end RAG: scaffold only.

## Artifacts

- `benchmarks/rag-evaluation/runs/scifact-full-test-v1/application-rrf-metrics-v4.json` (immutable 300-query metrics, per-query values, and bootstrap intervals)
- `benchmarks/rag-evaluation/runs/hakari-pilot-20260915-aggregate.json` and `.md` (three raw HAKARI task summaries)
- `benchmarks/rag-evaluation/runs/hakari-pilot-20260915-aggregate-v2.json` and `.md`
- `benchmarks/rag-evaluation/runs/hakari-public-comparison-v1.json` and `.md`
- `benchmarks/rag-evaluation/execution-status.json`
- `benchmarks/rag-evaluation/runs/plan.json`
