# Dedup experiment

Measures whether removing repeated candidates between RRF fusion and the reranker shortlist
improves hybrid retrieval. The app keeps its current behaviour: dedup only runs when
`RAG_HYBRID_DEDUP` is set (see `src/lib/server/rag/search/candidate-dedup.ts`).

| Variant | `RAG_HYBRID_DEDUP` | Drops a candidate when                                                  |
| ------- | ------------------ | ----------------------------------------------------------------------- |
| A       | `none` (default)   | never (current pipeline)                                                |
| B       | `exact`            | its lowercased, whitespace-normalized text matches an earlier candidate |
| C       | `near`             | it shares at least 90% of its 5-word shingles with an earlier candidate |
| D       | `document-cap`     | its document already has `RAG_HYBRID_DOC_CAP` (2) earlier candidates    |

All variants use RRF + Ettin 32M (100→40→20), default 1,200-character chunks and
nomic-embed-text v1.5. Datasets: SciFact, NFCorpus, a 300-query ArguAna sample (seed 42),
and `scifact-dup`, a SciFact copy with duplicated relevant documents built by
`make_scifact_dup.py`.

## Run

```bash
benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/make_scifact_dup.py
benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/run_experiment.py
benchmarks/beir/.venv/bin/python benchmarks/beir/dedup_experiment/analyze.py
```

`run_experiment.py` prepares the scifact-dup corpus copy if needed, then runs the 16
dataset × variant jobs one at a time (about two hours). It records run IDs in
`manifest.json`; running it again resumes where it stopped. `analyze.py` writes
`results.md` and `results.json`.

## Reading the results

Duplicated copies are judged relevant under their own IDs, so standard nDCG and Recall can
drop when dedup removes them even though the context gets more varied. Compare with
**distinct-relevant recall**, which counts each duplicate group once, and the duplicate rates.
Metrics are at document depth 5 and 10 from the harness's 20-chunk requests, not from
separate topK = 5 and topK = 10 pipeline runs.
