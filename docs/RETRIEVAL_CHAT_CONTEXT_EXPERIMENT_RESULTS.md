# Retrieval Chat-Context Experiment Results

Date: 2026-09-09  
Code revision: `844acd9d7d7cd48c752ec869f40c018608b11301`  
Experiment: `.cache/retrieval-experiments/20260909-chat-context-182627/`

## Outcome

The experiment did **not** find a ranking change that improved on the strongest baseline. BM25 was the strongest untuned validation method and the strongest final-test method. Validation selected fusion weight `1.00`, whose first 20 results were exactly BM25's results. It therefore tied BM25 on every final query while retaining the much higher cost of hybrid candidate generation and cross-encoding.

The saved learned model did not beat BM25. On the 50 fresh final-test queries it preserved at least one answer-bearing chunk in every five-chunk context, but it placed that chunk first only 76% of the time versus 100% for BM25. Its mean NDCG@5 was `0.726100`, versus `0.985905` for BM25: a difference of `-0.259805` with a paired whole-query bootstrap 95% interval of `[-0.296663, -0.224890]`.

These are synthetic-fixture results, not evidence of quality on a natural corpus. Every fresh query contains an exact fictional record code such as `SYN-063`, which strongly favors lexical retrieval. The result is still useful for this controlled objective: neither the current learned model nor rank fusion adds value over simply choosing BM25 for this workload.

## Artifacts and protocol

The reproducible artifact root has separate `runtime/`, `configuration/`, `judgments/`, `results/`, and `logs/` directories. Important files are:

- `configuration/protocol.json`: frozen before validation retrieval or fusion tuning.
- `configuration/current-path.json`: traced agent/tool/retrieval/context behavior.
- `configuration/queries.json`: regression, validation, and final-test queries with required facts.
- `configuration/frozen-selection.json`: timestamped validation choice, written before final retrieval.
- `configuration/saved-model.json`: copied model parameters and compatibility metadata.
- `configuration/run-experiment.mjs`: isolated retrieval, judgment, metric, and bootstrap runner.
- `configuration/run-answers.mjs`: no-tools controlled answer-generation runner.
- `runtime/app.db`: consistent backup of the prior synthetic database, created with Node SQLite's backup API.
- `runtime/previous-database-backup.json`: backup method and database fingerprint.
- `runtime/code-backup/`: starting working-tree/index patches and the relevant untracked report.
- `judgments/{regression,validation,final}.json`: blind-input synthetic judgments, rationales, provenance, and exact chunk text.
- `results/{regression,validation,final}.json`: every ranking, score, selected chunk, rating, exact formatted context, fact-survival check, confidence result, per-query metric, aggregate, and latency.
- `results/answers.json.partial`: checkpoint containing exact generated answers, contexts, reasoning traces, and automated checks. It has 23 complete query records; generation stopped while `FINAL-024` was in progress, so no completed `answers.json` exists.

The primary context limit was five. Each semantic and BM25 method had a fixed 20-result budget. Hybrid cross-encoded the deduplicated union of the first 20 semantic and first 20 BM25 candidates (at most 40), returned 20, and supplied the first five to context. The learned model reranked the 20 hybrid results with the saved model and current `0.35` learned-rank blend. Recall@20 is diagnostic; all primary metrics score only the five delivered chunks.

Fusion used the same at-most-40 candidate union. For a pool of `N`, a present rank `r` receives normalized score `(N-r)/(N-1)`; a missing BM25 rank receives zero. Fusion score is:

`BM25 weight × normalized BM25 rank + (1 − BM25 weight) × normalized cross-encoder rank`

Ties resolve by cross-encoder rank, BM25 rank, then chunk ID. The fixed grid was `0.00`, `0.25`, `0.50`, `0.75`, `1.00`; `0.00` is pure cross-encoder order and `1.00` is pure BM25 order for BM25-present candidates. Highest validation NDCG@5 won; exact ties favored the lowest BM25 weight.

Primary retrieval omitted confidence, matching the tool's behavior when the agent does not request it. Medium/high application filters were evaluated separately for score-compatible methods. Direct exact-query feedback adjustments were excluded. No evaluation label was written to retrieval feedback or training tables.

Metrics use gain zero for rating 1 and `2^(rating-1)-1` otherwise. Missing positions count as non-relevant for Precision@5, so its denominator is always five. This corpus has one answer-bearing chunk per new information need, making `0.20` the attainable Precision@5 ceiling; the metric does not demand five relevant chunks where only one exists.

## Current agent-to-context path

The current agent does not receive document context automatically. Its search-tool policy asks it to create a focused standalone `query`; `search.ts` trims that text to 2,000 characters. A valid tool `searchType` overrides the profile; otherwise the active profile retrieval mode applies, then the code default. Both the live active profile and code default are hybrid.

The tool's `top_k` overrides the profile when present. Otherwise the active profile uses `rag_top_k = 5`, matching the code default. On that normal five-result path, `retrieveRagContext` expands five to 15 candidates. Hybrid then obtains up to 15 semantic and 15 BM25 candidates after requesting 30 from each first-stage retriever, deduplicates the union, cross-encodes at most 30, and retains 15 before the optional learned rerank and final truncation to five.

Confidence is an optional tool argument, not a persisted profile setting. Omitted/low keeps all candidates. Semantic medium/high uses `0.55/0.70`; hybrid uses `0.20/0.50`; BM25 uses the larger of a top-score-relative threshold (`0.35/0.60`) and an absolute floor (`0.8/1.5`). Confidence filtering precedes exact-query feedback.

Only hybrid can use a learned model. The experimental setting must be enabled and an active compatible model must exist. The current live setting is off and no live model is active. A learned candidate gets 20 score/rank/presence features, a predicted utility, and a rank blend with weight `0.35`.

After final selection, `retrieveRagContext` collapses whitespace and limits each chunk to 1,200 characters. It formats chunks as `- <content> (source: <title-or-path>)` beneath `Relevant context:`. The search tool returns that exact block to the chatbot. The controlled answer check supplied the captured block through the no-search document prompt, followed by `Request: <question>`, so generation could not retrieve different evidence.

## Isolation and starting-point reproduction

The repository began at the revision above with one unrelated untracked file, `docs/RETRIEVAL_SYNTHETIC_EXPERIMENT_RESULTS.md`. It was copied into the experiment backup. Nothing was staged, committed, reset, cleaned, or rebased.

The prior database was backed up from `.cache/retrieval-experiments/20260909-synthetic/runtime/app.db` into the new runtime before retrieval. The runner requires `process.cwd()` to equal the isolated runtime before importing model/database libraries, opens only `runtime/app.db` with `readOnly: true`, enables SQLite `query_only`, and refuses artifact writes outside the new experiment directory. Shared cached embedding and reranker files were read with remote model access disabled.

The backup contains the expected 120 documents, 599 chunks, 2,000 synthetic ratings, 130 impressions, saved run `21a21ce5-ef94-479f-bea0-aa44ed961694`, and saved model `ac4d6105-59e8-48d0-9491-e0aa57871dcc`. The model is dataset version 4, feature version 2, embedding model `nomic-ai/nomic-embed-text-v1.5`, reranker `Xenova/ms-marco-MiniLM-L-6-v2`, scoring version `retrieval-signals-v1`, and ranking strategy `pairwise-logistic-rank-blend-v3`. Its 20 feature names, scaler values, weights, and intercept match the saved artifact and are finite.

Reproduction initially revealed that the old benchmark had expanded requested top-20 retrieval to 60 candidates before hybrid/learned reranking. After restoring that historical budget for reproduction only, all 30 semantic, BM25, hybrid, and learned top-20 rankings matched the old artifacts position-for-position. Because rankings and immutable judgments are exact, the old reported metrics reproduce with zero tolerance difference:

| Historical method | Recall@20 |   MRR@20 |   NDCG@5 |
| ----------------- | --------: | -------: | -------: |
| Semantic          |  0.933333 | 0.633836 | 0.644262 |
| BM25              |  1.000000 | 1.000000 | 0.878827 |
| Hybrid            |  1.000000 | 0.983333 | 0.817691 |
| Learned           |  1.000000 | 0.983333 | 0.824659 |

The controlled validation/final candidate budget remained the protocol's fixed 20; the historical 60-candidate reproduction did not change it.

## Query construction and judging

The original 30 held-out queries are a regression set, not an untouched test set. The new set has 30 validation and 50 final-test information needs. Validation contains ten assignment, ten milestone, and ten record-series needs. Final contains 17 assignment, 17 milestone, and 16 record-series needs.

Each new need uses a unique `(document, fact type)` pair absent from both original query sets, and none of the 80 new needs reuse a document. Normalized query strings are also disjoint. They are genuinely different canonical facts, not paraphrases of an existing need, but use only three fixed question templates and exact `SYN-nnn` identifiers. This reduces linguistic variety and is an important external-validity limitation.

Judgments were generated from actual candidate chunk text with method, rank, and score excluded from the judgment input. A chunk containing every requested canonical sentence is rated 5; a partial answer is 4; a correct-record chunk without the answer is 3; a wrong-record chunk sharing an answer value is 2; otherwise it is 1. Every row includes a short rationale and explicit synthetic provenance. There were no rating-4 cases because every new need required one canonical sentence that appeared whole in a single chunk.

The validation pool has 1,073 judgments: 731/221/91/0/30 at ratings 1/2/3/4/5. The final pool has 1,824: 1,257/363/154/0/50. Pools include every candidate introduced by every compared ranking. No unjudged item was assigned a default relevance, and no evaluation judgment entered training feedback.

## Validation selection

| Method/weight        | Validation NDCG@5 |
| -------------------- | ----------------: |
| Semantic             |          0.483156 |
| BM25                 |          0.981948 |
| Hybrid / fusion 0.00 |          0.753202 |
| Learned              |          0.756243 |
| Fusion 0.25          |          0.798947 |
| Fusion 0.50          |          0.815652 |
| Fusion 0.75          |          0.869076 |
| Fusion 1.00          |          0.981948 |

BM25 was frozen as the strongest untuned baseline. Fusion `1.00` was frozen before final-test retrieval. Its equality with BM25 is structural, not an observed post-test adjustment: the candidate union contains the entire BM25 top 20, and pure BM25 rank puts those 20 ahead of every semantic-only candidate.

## Fresh final retrieval results

All 50 final queries are answerable. Full per-query values and exact contexts are in `results/final.json`.

| Method               | Context hit | Top-1 relevant | Precision@5 |   NDCG@5 | Fact coverage | Harmful count/query | Harmful rate | Recall@20 |
| -------------------- | ----------: | -------------: | ----------: | -------: | ------------: | ------------------: | -----------: | --------: |
| Semantic             |       0.720 |          0.420 |       0.144 | 0.492756 |         0.720 |                3.88 |        0.776 |     0.920 |
| BM25                 |       1.000 |          1.000 |       0.200 | 0.985905 |         1.000 |                0.94 |        0.188 |     1.000 |
| Hybrid               |       1.000 |          0.760 |       0.200 | 0.718442 |         1.000 |                3.76 |        0.752 |     1.000 |
| Saved learned        |       1.000 |          0.760 |       0.200 | 0.726100 |         1.000 |                3.74 |        0.748 |     1.000 |
| Selected fusion 1.00 |       1.000 |          1.000 |       0.200 | 0.985905 |         1.000 |                0.94 |        0.188 |     1.000 |

Relative to BM25, learned ranking changed mean context hit, Precision@5, fact coverage, and Recall@20 by zero, but changed top-1 relevance by `-0.24`, NDCG@5 by `-0.259805`, harmful chunks per query by `+2.80`, and harmful-context rate by `+0.56`. It degraded NDCG on all 50 queries, improved none, tied none, and had worst change `-0.587792`. It did not lose all relevant context because the answer-bearing chunk remained within five.

Selected fusion changed every aggregate by exactly zero. All 50 queries tied, its worst change was zero, and its paired bootstrap interval is `[0, 0]`. Learned's paired interval is `[-0.296663, -0.224890]`. Both use 10,000 whole-query resamples with seed `20260909`; chunks were not resampled independently.

Learned ranking did add a small amount over hybrid itself: mean final NDCG@5 `+0.007659`, with 12 improved, one degraded, and 37 tied queries; worst change was `-0.019274`. That does not make it competitive with BM25.

## Confidence, truncation, and integration findings

With confidence omitted, all methods deliver five chunks. On final BM25, medium confidence dropped 14.72 of 20 candidates on average, retained 100% context hit/top-1/fact coverage, reduced harmful-context rate from `0.188` to `0.004`, and produced NDCG@5 `0.979591`. High confidence dropped 18.42, still retained every answer, removed all harmful chunks, and produced NDCG@5 `0.799039`; its lower NDCG reflects missing lower graded positions being counted as zero, not loss of the required fact.

Hybrid medium/high retained every answer but did not improve top-1 relevance (`0.760`); harmful rates were `0.754/0.726`. Semantic medium/high retained the same results for this score distribution and still missed answer evidence in 14 top-five contexts. Four semantic queries lacked an answer-bearing chunk even at rank 20.

The learned output score is a blend of normalized ranks, not a cross-encoder probability. Applying the application's hybrid `0.20/0.50` thresholds to it removed candidates but did not change the first-five metrics here; those thresholds have no calibrated probabilistic meaning for learned scores. Fused ranks have the same incompatibility, so the experiment did not apply raw hybrid thresholds to fusion. Production integration would need a defined confidence policy rather than reusing an unrelated score threshold.

No selected final chunk exceeded the 1,200-character delivery limit after whitespace compaction, so no answer fact was truncated. This corpus therefore tests ranking and filtering but provides no positive evidence about handling genuinely overlength answer passages.

## Latency after warm-up

Final-test mean timings were approximately:

| Operation                                                      | Mean milliseconds/query |
| -------------------------------------------------------------- | ----------------------: |
| Semantic retrieval                                             |                   12.31 |
| BM25 retrieval                                                 |                    1.51 |
| Cross-encoder stage                                            |                  827.77 |
| Hybrid end-to-end estimate with parallel first-stage retrieval |                  840.08 |
| Learned reranking increment                                    |                    0.28 |
| Fusion arithmetic increment                                    |                    0.01 |

The selected fusion still needs the 840 ms hybrid pipeline before its negligible arithmetic step, yet returns BM25's roughly 1.5 ms ranking. It is therefore materially worse than directly selecting BM25 even though retrieval metrics tie.

## Failure inspection

The dominant failure is cross-encoder ordering among highly repetitive synthetic passages, not candidate absence. BM25's exact record code put the answer first on every final query. Hybrid always had the answer in its pool and top five, but put it at ranks 1/2/3/4 on 38/9/2/1 queries.

- `VAL-002`, asking for `SYN-063` review/transfer facts: BM25 put answer chunk `6eecd36d…a967e` first. Hybrid and learned put unrelated 1-star `542f241e…712af3c` (`SYN-053`, “amber review in 1957…”) first and the correct chunk second. The cross scores were extremely close (`0.988680` versus `0.988151`), indicating confusion over shared template language and nearby periods.
- `VAL-016`, asking for `SYN-069` organization/location/year: BM25 put answer `37cee993…fbbd` first. Hybrid put 2-star `cd505a54…e416d` (`SYN-049`, another assignment passage) first, 1-star `SYN-059` second, and the answer third. Learned moved the answer to second but still left the keyword-heavy wrong entity first.
- `VAL-029`, asking for `SYN-031` review/transfer facts: semantic and BM25 both put answer `d1f87a71…96173` first. Cross-encoding instead put 1-star `884f0564…f5f52` (`SYN-041`) first by a probability margin of only about `0.000015`; learned preserved that error.
- `VAL-011`, asking for `SYN-064` review/transfer facts: the answer stayed first, but hybrid filled the remaining leading context with wrong-record passages sharing “review” and years. BM25 instead placed three same-record, non-answer chunks after the answer. This explains much of the harmful-context gap.
- Semantic candidate retrieval missed the answer within 20 for final cases `FINAL-003`, `FINAL-021`, `FINAL-036`, and `FINAL-045`, and missed it within five for 14 cases. Hybrid recovered all of these through the BM25 branch.

No validation or regression case showed the learned model demoting a 4–5-star chunk relative to hybrid; its regressions came from reordering lower-rated material. It improved hybrid slightly in some cases, but its score/rank-only features and small pairwise linear capacity did not encode an explicit exact-entity match strong enough to recover BM25's advantage. More ratings alone would not fix missing semantic candidates or guarantee correction of cross-encoder entity/period confusion.

The controlled comparisons support these diagnoses:

- **Candidate retrieval:** semantic has misses; BM25 and the hybrid union do not on this workload.
- **Ranking:** cross-encoder ordering is the main source of lost top-1 quality and added harmful context.
- **Learned features/capacity:** current features can make small rank corrections but do not match the exact-identifier signal already present in BM25.
- **Filtering/truncation:** optional BM25 filtering can reduce harmful material without losing answers here; learned/fusion score semantics are not compatible with existing thresholds. Truncation was inactive in all selected chunks.
- **Answer generation:** reported below from the controlled local-model run; retrieval evidence, not generation, is the primary experimental difference.

## Controlled chatbot answers

The controlled answer sample is the first 20 final queries, selected before inspecting answer outcomes. It is balanced across the frozen query construction: 7 assignment, 7 milestone, and 6 record-series needs. The sample uses the exact saved BM25, selected-fusion, and learned contexts from `results/final.json`; selected fusion and BM25 contexts are identical. BM25 and learned answers used the same local Gemma model, application document-context prompt, tools disabled, temperature `0.2`, top-k `8`, visible token budget `1024`, reasoning budget `512`, and seed `20260909`. Fusion reuses the identical BM25 answer because its captured context is identical. The remaining 30 final queries were not answer-tested.

The checkpoint contains exact answers and contexts for this 20-query sample (plus three additional complete records retained as an incomplete-run artifact). Answer judging is synthetic and automated: canonical values, source naming, unsupported domain values, and abstention behavior were checked against fictional ground truth and delivered context. It is not human judging.

| Method               | Correct required facts | Unsupported-claim rate | Source-support rate | Appropriate-abstention rate | Mean generation latency |
| -------------------- | ---------------------: | ---------------------: | ------------------: | --------------------------: | ----------------------: |
| BM25                 |          20/20 (1.000) |           0/20 (0.000) |       20/20 (1.000) |               20/20 (1.000) |             47,579.5 ms |
| Learned              |          20/20 (1.000) |           0/20 (0.000) |       20/20 (1.000) |               20/20 (1.000) |             45,711.8 ms |
| Selected fusion 1.00 |          20/20 (1.000) |           0/20 (0.000) |       20/20 (1.000) |               20/20 (1.000) |     same answer as BM25 |

Within this limited synthetic sample, answers did not improve: all methods passed every automated check, and fusion was necessarily identical to BM25. The partial file remains the authoritative checkpoint; it must not be described as a 50-query answer evaluation.

## Acceptance criteria

| Criterion                                     | BM25        | Learned                    | Selected fusion    |
| --------------------------------------------- | ----------- | -------------------------- | ------------------ |
| Context hit ≥ 95%                             | PASS (100%) | PASS (100%)                | PASS (100%)        |
| Top-1 relevance ≥ 90%                         | PASS (100%) | FAIL (76%)                 | PASS (100%)        |
| Mean NDCG gain over strongest baseline ≥ 0.02 | baseline    | FAIL (-0.2598)             | FAIL (0)           |
| Paired 95% interval above zero                | baseline    | FAIL (entirely below zero) | FAIL ([0, 0])      |
| No aggregate context-hit reduction            | baseline    | PASS                       | PASS               |
| Non-tie win rate ≥ 0.60                       | baseline    | FAIL (0/50)                | FAIL (no non-ties) |
| Worst NDCG change ≥ -0.02                     | baseline    | FAIL (-0.5878)             | PASS (0)           |

Meeting basic context quality does not establish added value. Learned ranking meets the hit-rate target but fails top-1 and every improvement criterion. Fusion meets basic quality only because it duplicates BM25; it demonstrates no improvement.

## Isolation verification

The live database still has size `403099648` bytes and SHA-256 `6413031283cfcd02567fe2e6f607a4dbfeb520aaa9774e228f838079f2251885`, matching `configuration/live-before.json`. The live experimental setting remained off and no live model was activated. The isolated runner opened the copied database read-only with SQLite `query_only`; the answer runner only read retrieval artifacts and the shared local model. No evaluation judgments entered live feedback, impressions, or training tables. The source synthetic experiment remains unchanged, and no experiment process remained running at audit time. No files were staged or committed.

## Recommendation

Do not activate the saved learned model and do not propose the selected fusion as production behavior. Weight `1.00` is BM25 with avoidable cross-encoder cost.

If a separate production proposal is desired, the defensible proposal is narrower: improve retrieval-mode selection so exact identifiers/proper names reliably choose the existing BM25 path, or evaluate an explicit exact-entity feature/gate before cross-encoder reranking on a more natural, independently judged corpus. This experiment does not justify changing the default hybrid mode globally, activation thresholds, or confidence policy.
