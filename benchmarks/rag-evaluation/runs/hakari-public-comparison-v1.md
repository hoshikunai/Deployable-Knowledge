# HAKARI public comparison v1

Public source: `benchmarks/rag-evaluation/runs/hakari-public-comparison-v1/source/duckdb/hakari_bench.duckdb`
Dataset revision: `d3962aa8efe48ed79044c5e155b848982667b4ba`
Filters: model_type=reranker, candidate_ranking=reranking_hybrid, score_target=reranking

| Task | Local nDCG@10 | Comparable systems | Rank / ties | Percentile | Public median | Best |
|---|---:|---:|---:|---:|---:|---:|
| arguana | 0.384292 | 1 | 2 / 2 | 0.0% | 0.387587 | 0.387587 |
| scifact | 0.734496 | 1 | 2 / 2 | 0.0% | 0.804590 | 0.804590 |
| nfcorpus | 0.361286 | 12 | 5 / 5 | 66.7% | 0.353372 | 0.424094 |
| fiqa2018 | 0.507949 | 1 | 2 / 2 | 0.0% | 0.581443 | 0.581443 |

Macro local nDCG@10: **0.497006**
Macro public median: **0.531748**

These are directly filtered NanoBEIR task comparisons. The local system is not a Hugging Face model despite the raw HAKARI metadata using source type `huggingface`; it is a custom local TypeScript bridge. The existing raw JSON is immutable, so this limitation is documented rather than rewritten.
