# Synthetic Retrieval Training Experiment Results

**Synthetic test fixture only.** All documents, canonical facts, queries, and ratings in this experiment were generated for an isolated test. The ratings are AI-generated test fixtures, not human-authored evidence and not evidence about the live corpus.

## Environment

- Experiment: `.cache/retrieval-experiments/20260909-synthetic/`
- Runtime/database: `.cache/retrieval-experiments/20260909-synthetic/runtime/`
- Corpus: `.cache/retrieval-experiments/20260909-synthetic/corpus/`
- Ground truth: `.cache/retrieval-experiments/20260909-synthetic/ground-truth/`
- Ratings: `.cache/retrieval-experiments/20260909-synthetic/ratings/`
- Results: `.cache/retrieval-experiments/20260909-synthetic/results/`
- Fixed seed: `20260909`
- Setting: `experimentalRetrievalTrainingEnabled`, enabled only in the isolated instance
- Dataset/feature versions: 4 / 2
- Training run/model: `21a21ce5-ef94-479f-bea0-aa44ed961694` / `ac4d6105-59e8-48d0-9491-e0aa57871dcc`

## Corpus and ratings

The corpus contained 120 synthetic aviation-themed documents, 599 actual indexed chunks, and 768-dimensional embeddings. All 120 documents were accepted through the real ingestion endpoint. Training used 100 distinct information needs and held out 30 distinct evaluation information needs.

The 100 training searches returned 20 displayed hybrid results each. All 2,000 ratings were submitted through the human chunk-rating endpoint with real impression-result IDs; no hidden candidates were rated through that endpoint. The rating distribution was 2: 878, 3: 950, 4: 72, and 5: 100. Candidate capture covered all 100 training impressions, with 40 hybrid candidate rows per impression and valid attribution for every rating.

## Training readiness and gates

Training completed with 2,000 compatible/attributed ratings, 100 distinct training queries, 7,468 training preference pairs, and 100 evaluable hybrid groups. Held-out query hashes were absent from training feedback, and all model inputs were finite.

The trainer reported pairwise accuracy `0.774238`, baseline NDCG@5 `0.833305`, trained NDCG@5 `0.835302`, and training NDCG improvement `0.001997`. The application activation gates therefore were: evaluated groups PASS (100), NDCG improvement FAIL (0.001997 < 0.005), non-tie win rate FAIL (10/18 = 0.556 < 0.60), worst-group regression FAIL (-0.093430 < -0.02), and parameter/compatibility checks PASS. The model was not activated.

## Held-out evaluation

Metrics below use the same 30 held-out queries, candidate generation, relevance judgments, and metric formulas. Learned ranking was evaluated offline with the saved model parameters, the application’s feature construction/scaler/utility prediction, and its 0.35 blend. Full details are in `results/offline-evaluation.json`.

| Ranking                   | Recall@20 |   MRR@20 |   NDCG@5 |
| ------------------------- | --------: | -------: | -------: |
| Semantic                  |  0.933333 | 0.633836 | 0.644262 |
| BM25                      |  1.000000 | 1.000000 | 0.878827 |
| Hybrid baseline           |  1.000000 | 0.983333 | 0.817691 |
| Saved learned model       |  1.000000 | 0.983333 | 0.824659 |
| Oracle judged hybrid pool |  1.000000 | 1.000000 | 1.000000 |

The learned-vs-hybrid changes were Recall `0`, MRR `0`, and NDCG@5 `+0.006968`. Seven queries improved, two degraded, and 21 tied; the worst NDCG change was `-0.037116`. A paired bootstrap over whole queries using 10,000 resamples and the fixed seed produced a 95% interval of `[-0.001112, 0.016688]` for mean NDCG change. The interval includes zero.

The oracle ceiling of 1.0 NDCG shows that relevant chunks were available in the judged hybrid pools. The baseline was already strong, but some ordering headroom remained; this controlled result does not establish that the learned model generalizes to real data.

## Isolation verification

The experiment database was created fresh by replaying the unchanged generated migrations. The live database was never used for ingestion, search, ratings, training, or activation. The live setting remained OFF, its active model remained null, and its post-cleanup human-only records were unchanged during the experiment. The isolated database contains 120 documents, 599 chunks, 2,000 synthetic human-source test ratings, 130 search impressions, and no active model.

The disabled-toggle check returned ordinary search results without impression-result IDs and left the isolated stored ratings intact; re-enabling did not activate a model. The only process started for this experiment was the isolated server on `127.0.0.1:4179` and it has been stopped.

## Limitations and conclusion

The corpus, relevance labels, and ratings are synthetic; the query split is fixed but small; the learned model failed the application’s activation gates; and the confidence interval includes zero. The rating heuristic was designed to simulate a knowledgeable evaluator against the canonical facts, not to claim human judgment. The experiment demonstrates that the ingestion, capture, human-rating, training, persistence, and offline evaluation pipeline functions end to end, but **improvement inconclusive** on this synthetic test.
