# HAKARI reranker pilot (v2)

Package: hakari-bench 0.1.0 / commit `e2feac3614a738b17dae5cdb6066d152e4fe3f1c`
Model: TypeScript `Xenova/ms-marco-MiniLM-L-6-v2` via loopback sidecar
Candidate ranking: `reranking_hybrid`

| Dataset | nDCG@10 | Acc@100 | Runtime (s) |
|---|---:|---:|---:|
| NanoArguAna | 0.384292 | 1.000000 | 275.73 |
| NanoFiQA2018 | 0.507949 | 0.980000 | 227.11 |
| NanoNFCorpus | 0.361286 | 1.000000 | 243.99 |
| NanoSciFact | 0.734496 | 1.000000 | 250.43 |

Macro nDCG@10: **0.497006**

All four tasks completed through the real local TypeScript scorer. These are NanoBEIR reranker results, not full BEIR scores.

Coverage and candidate relevant-coverage fields are retained in each raw HAKARI JSON artifact; HAKARI reports Acc@100 as the available candidate-coverage measure.
