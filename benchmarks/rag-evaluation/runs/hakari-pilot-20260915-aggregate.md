# HAKARI reranker pilot

Package: hakari-bench 0.1.0 (e2feac3614a738b17dae5cdb6066d152e4fe3f1c)
Model bridge: TypeScript `Xenova/ms-marco-MiniLM-L-6-v2`
Candidate ranking: `reranking_hybrid`

| Dataset | nDCG@10 | Acc@100 | Seconds |
|---|---:|---:|---:|
| NanoFiQA2018 | 0.507949 | 0.980000 | 227.11 |
| NanoNFCorpus | 0.361286 | 1.000000 | 243.99 |
| NanoSciFact | 0.734496 | 1.000000 | 250.43 |

Macro nDCG@10: **0.534577**

These are HAKARI NanoBEIR reranker results using fixed `reranking_hybrid` candidates. They are not full BEIR scores and should not be compared numerically with the application SciFact full-corpus result.

Raw artifacts are referenced in the aggregate JSON.
