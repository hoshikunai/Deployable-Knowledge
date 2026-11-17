# Fusion experiment

Tests four search-time switches. Title in reranker input became the app default after this
experiment; the others stay off. The experiment pins every setting, so re-runs are unaffected.

| Setting                 | Env var                                          | Default | What it changes                                                                      |
| ----------------------- | ------------------------------------------------ | ------- | ------------------------------------------------------------------------------------ |
| RRF weights             | `RAG_RRF_SEMANTIC_WEIGHT`, `RAG_RRF_BM25_WEIGHT` | 1, 1    | How much each retriever's ranks count in fusion                                      |
| Fusion level            | `RAG_RRF_LEVEL`                                  | `chunk` | `document`: agreement on a document through different chunks counts                  |
| Document order          | `RAG_HYBRID_DOCUMENT_ORDER`                      | `none`  | `sum-top2`: rank documents by their two best chunks, keeping their chunks together   |
| Title in reranker input | `RAG_RERANK_TITLE`                               | `true`  | Prepend the document title to chunks that don't start with it, for the reranker only |

To avoid fitting settings to the reported queries, fusion settings are tuned on
**SciFact train** (809 queries) and **NFCorpus dev** (324), then reported once on **test**.
ArguAna has only a test split, so it is reported with the settings chosen elsewhere
(300-query sample, seed 42).

## Run

```bash
benchmarks/beir/.venv/bin/python -u benchmarks/beir/fusion_experiment/run_experiment.py
benchmarks/beir/.venv/bin/python benchmarks/beir/fusion_experiment/analyze.py
```

1. **Tune (RRF only):** five semantic:BM25 weight pairs (1:1, 1:0.75, 1:0.5, 0.75:1, 0.5:1),
   then fusion level × document order at the pair with the best average nDCG@10.
2. **Test:** per dataset, RRF baseline and tuned; RRF + Ettin baseline, with title, tuned, and
   tuned with title. Nothing is chosen on test.

About 2 hours. Run IDs are cached in `manifest.json` by configuration; running again resumes.
`analyze.py` writes `results.md` and `results.json`, including average chunks per top-10
document (checks whether `sum-top2` favours long documents) and the share of top-10 documents
found by both retrievers.

The test baselines use the default settings, so they double as a regression check: SciFact
should reproduce 0.73377 (RRF only) and 0.73252 (RRF + Ettin).
