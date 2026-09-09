# Retrieval Training Results

Date: 2026-09-08  
Code revision: `c339ba8d29838260403f7e3bacd2491d5d96b165`  
Dataset version: 4  
Feature version: 2  
Ranking strategy: `pairwise-logistic-rank-blend-v3`

## Artifacts

- Backup: `/tmp/dk-retrieval-training.CX6nBOGy`
- Raw human response: `/tmp/dk-retrieval-training.CX6nBOGy/human-response.json`
- SQLite backup was created with Node's SQLite backup API at `/tmp/dk-retrieval-training.CX6nBOGy/app.db`.

## Data readiness

| Feedback source | Total ratings | Attributed and compatible | Distinct normalized queries | Rating distribution (1/2/3/4/5) | Within-impression preference pairs | Groups with hidden candidates | Readiness |
| --------------- | ------------: | ------------------------: | --------------------------: | ------------------------------- | ---------------------------------: | ----------------------------: | --------- |
| `human_expert`  |           304 |                       303 |                          43 | 21 / 38 / 66 / 88 / 91          |                              1,270 |                             0 | Ready     |

The human source met the minimums of 30 compatible ratings, 10 distinct queries, and training preferences in every cross-validation fold. The historical groups available for training contain displayed results only; no missing candidates were fabricated.

## Human-expert model

- Run ID: `7a094734-a814-4dc6-8509-eca1c1dd9d50`
- Model ID: `b6a4c89c-d64a-48a4-bd79-5fbbb9e42e93`
- Status: completed
- Training examples / preference pairs: 303 / 1,270
- Distinct queries / cross-validation folds: 43 / 5
- Pairwise accuracy: 0.653900
- Baseline NDCG@5: 0.829136
- Trained NDCG@5: 0.830209
- NDCG improvement: 0.001073
- Evaluated hybrid ranking groups: 35
- Improved / degraded / tied groups: 5 / 2 / 28
- Non-tie win rate: 0.714286
- Worst group NDCG change: -0.030498

Activation gates:

- Evaluated hybrid groups ≥ 10: **PASS** (35)
- NDCG improvement ≥ 0.005: **FAIL** (0.001073)
- Non-tie win rate ≥ 0.60: **PASS** (0.714286)
- Worst group NDCG change ≥ -0.02: **FAIL** (-0.030498)
- Compatibility and parameter validation: **PASS** (dataset/features/strategy compatible; 20 finite weights and positive finite scaler standard deviations)

Result: not eligible for activation.

## Activation state and limitations

No activation request was made. `app_state.active_retrieval_model_id` was `null` before training and remains `null` after training.

The evaluation is query-disjoint cross-validation over rated candidates and does not establish improvement on the independent fixed benchmark. No independent benchmark comparison was run, so no benchmark improvement is claimed. The historical groups contain only displayed results, which limits candidate-context coverage.

Repository checks completed: `npm run lint`, `npm run check` (0 errors, 0 warnings), `npm run build`, and `git diff --check`. The build emitted only its existing large-chunk and adapter-auto environment notices.
