# AI-Proxy Judged Benchmark — Fixed Baseline Results

Date: 2026-09-09
Code revision: `c339ba8d29838260403f7e3bacd2491d5d96b165`
Explicit label: AI-proxy judged benchmark

## Validation

All 416 proxy judgments passed validation. The 12 cases contain no case without a 4- or 5-star candidate.

- Overall rating distribution: 1★ 118, 2★ 77, 3★ 67, 4★ 103, 5★ 51
- Recorded reviewerModel: `GPT-5`
- Provenance limitation: this reviewerModel value was preserved from the audit file; its identity was not independently verified.

## Artifacts

- Benchmark inputs: `.cache/retrieval-benchmarks/20260908-broad-corpus/`
- This run: `.cache/retrieval-benchmarks/20260908-broad-corpus/proxy-baseline/20260909T124249Z/`
- Validation summary: `proxy-baseline/20260909T124249Z/validation-summary.json`
- Frozen-input validation: `proxy-baseline/20260909T124249Z/frozen-input-validation.json`
- Disposable evaluation database: `proxy-baseline/20260909T124249Z/evaluation-runtime/app.db`
- Complete report JSON: `proxy-baseline/20260909T124249Z/baseline-report.json`
- Backup metadata and patches: `proxy-baseline/20260909T124249Z/`

The original proxy ratings and earlier `baseline-rankings.json` were not overwritten.

## Input fingerprints

Full SHA-256 records are in `proxy-baseline/20260909T124249Z/input-sha256.txt`.

| Input                     | SHA-256                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `queries.json`            | `2469c6c1ae91bfb30fcec333be6f1e857332825ae67746dd68240b42e8ba9ffe` |
| `judging-manifest.json`   | `f9ade97ba448c27a7fcf03acd77cf724e5dee4e03614fdd2ea7798259d052413` |
| `proxy-judgments.csv`     | `ef1bfad0037dd627c75a5db025c335f59f212a9ee3e1b9d4131eb0a97e8d2da4` |
| `proxy-rating-audit.json` | `55087ab5097b44393b88c12928afcf7103cda351a18c1c536b7998cdbe365d4c` |
| `capture-metadata.json`   | `441cd2e6599b899da3a0b914fd6f68330ab42492f05d0c723ea5cf7de1f1d88a` |
| `model-artifacts.json`    | `59bb6a9587974cdc5cf679f86ac2f677234013abcce047941dd50979713b5417` |
| `runtime.mjs`             | `9c8f93935c53db5df262838baeb900b7171e3366887a056d18c5c0b5d66f7b9e` |
| `package-lock.json`       | `4ba95af9bb0227ae242815213899046f652252db78d8769618dee3307a3aceb2` |

The frozen corpus fingerprint matched capture metadata: 11 documents and 5,370 chunks. All judged chunks, cached model files, runtime, package lock, and query-set fingerprints matched.

## Benchmark configuration

- Cases: 12
- Judgments: 416
- Top K: 20
- Corpus: 11 active documents, 5,370 chunks
- Embedding model: `nomic-ai/nomic-embed-text-v1.5`
- Reranker model: `Xenova/ms-marco-MiniLM-L-6-v2`
- Scoring version: `retrieval-signals-v1`
- Direct feedback adjustments: disabled
- Active learned model: none (`activeModelId: null`)
- Persisted disposable-copy result: 12 cases and 416 judgments

## Aggregate baseline metrics

| Retrieval method | Recall@20 | Mean reciprocal rank@20 |   NDCG@5 |
| ---------------- | --------: | ----------------------: | -------: |
| Semantic         |  0.654140 |                0.782738 | 0.505273 |
| BM25             |  0.737597 |                0.895833 | 0.617898 |
| Hybrid baseline  |  0.746653 |                0.836111 | 0.619693 |

Hybrid baseline performed best on Recall@20 and NDCG@5. BM25 performed best on mean reciprocal rank@20.

With no active model, the implementation's `hybridLearned` metrics exactly matched `hybridBaseline`; this is not evidence about a trained model.

## Per-query results

Values are Recall@20 / MRR@20 / NDCG@5.

| Query                             | Semantic                       | BM25                           | Hybrid baseline                |
| --------------------------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| 01 Command archive discovery      | 0.692308 / 1.000000 / 0.431679 | 0.653846 / 1.000000 / 0.569324 | 0.576923 / 0.500000 / 0.431679 |
| 02 Research collection navigation | 0.727273 / 1.000000 / 0.520098 | 0.727273 / 1.000000 / 0.720996 | 0.909091 / 1.000000 / 0.681093 |
| 03 Enlisted roles                 | 1.000000 / 0.500000 / 0.683891 | 1.000000 / 1.000000 / 0.882344 | 1.000000 / 1.000000 / 0.938018 |
| 04 Enlisted force structure       | 0.500000 / 0.142857 / 0.048182 | 0.500000 / 0.250000 / 0.222647 | 1.000000 / 0.333333 / 0.367193 |
| 05 T-38 evaluation grading        | 0.520000 / 1.000000 / 0.650828 | 0.480000 / 1.000000 / 0.800179 | 0.720000 / 1.000000 / 0.720663 |
| 06 T-38 mission preparation       | 0.500000 / 1.000000 / 0.763416 | 0.777778 / 1.000000 / 0.863670 | 0.666667 / 1.000000 / 0.909557 |
| 07 Pilot training techniques      | 0.454545 / 0.500000 / 0.439192 | 0.909091 / 0.500000 / 0.363513 | 0.636364 / 0.200000 / 0.242691 |
| 08 Personnel record correction    | 0.500000 / 0.250000 / 0.228609 | 0.714286 / 1.000000 / 0.477199 | 0.428571 / 1.000000 / 0.518889 |
| 09 Uniform accessories            | 0.666667 / 1.000000 / 0.591271 | 0.666667 / 1.000000 / 0.725008 | 0.666667 / 1.000000 / 0.591271 |
| 10 Publications management        | 0.888889 / 1.000000 / 0.481488 | 0.888889 / 1.000000 / 0.518259 | 0.888889 / 1.000000 / 0.814258 |
| 11 Decoration timing              | 0.800000 / 1.000000 / 0.592341 | 0.600000 / 1.000000 / 0.542961 | 0.666667 / 1.000000 / 0.597296 |
| 12 Casualty documentation         | 0.600000 / 1.000000 / 0.632284 | 0.933333 / 1.000000 / 0.728672 | 0.800000 / 1.000000 / 0.623708 |

## Limitations and state preservation

These judgments were generated by an AI reviewer and are not independent human validation. Recall is measured against pooled, judged relevant chunks, not every potentially relevant chunk in the corpus. These are baseline results only; no learned-model improvement has been tested. The benchmark judgments remain held out from training.

The live database remained unchanged: human feedback (304), AI-proxy feedback (50), retrieval impressions (120), impression results (3,706), saved training runs (10), saved models (10), benchmark cases (0), benchmark judgments (0), and active-model state (`null`). The 12 cases and 416 judgments exist only in the disposable evaluation copy.

No model was trained or activated during this task. No saved-model comparison was performed.
