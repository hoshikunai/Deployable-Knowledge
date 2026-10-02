# Fusion experiment: findings and recommendation

Run 2026-09-29/30 on `feature/beir-harness`; 34 runs, all complete. Full tables: `results.md`
(from `analyze.py`). Pipeline: 100→40→20, 1,200-character chunks, nomic-embed-text v1.5.
Fusion settings were tuned on SciFact train (809 queries) and NFCorpus dev (324) and reported
once on test. The test baselines reproduce earlier runs exactly (SciFact 0.73377 RRF only and
0.73252 with Ettin; NFCorpus 0.35447 and 0.34269), so the new code does not change defaults.

## Recommendation

| Setting                 | Recommendation                       | Evidence                                                                  |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------- |
| RRF weights             | Keep 1 : 1                           | Best tuned pair (semantic 1 : BM25 0.5) did not carry over to test        |
| Document-level fusion   | Leave off                            | Slightly worse than chunk fusion in tuning (−0.0016 average nDCG@10)      |
| Order by document       | Leave off                            | Large losses in tuning (−0.09 to −0.14 average nDCG@10)                   |
| Title in reranker input | Optional with a reranker; not proven | Never hurt, but no significant gain on the datasets that have real titles |

## RRF weights

Down-weighting BM25 to 0.5 was the best of five pairs on train/dev (average nDCG@10 0.5405 vs
0.5350 at 1 : 1). On test it was slightly worse everywhere with RRF only (SciFact −0.0016,
NFCorpus −0.0005, ArguAna −0.0114, p = 0.10) and mixed with Ettin (−0.006, −0.006, +0.003).
A gain that doesn't transfer from train/dev to test is tuning noise. ArguAna in particular
favours BM25, so down-weighting it hurts there.

## Document scoring

- **Document-level fusion** (agreement credit across different chunks of one document) was
  marginally worse than chunk fusion at the chosen weights, so it was not taken to test.
- **Ordering by document (sum of top 2 chunk scores)** collapsed rankings in tuning: SciFact
  train 0.741 → 0.592, NFCorpus dev 0.340 → 0.305. A single-chunk document can only contribute
  one score to the sum, so any document with two decent chunks outranks it: on SciFact train the
  top-ranked document went from single-chunk 30% of the time to never, and was relevant 43% of
  the time instead of 63%. 32% of SciFact's relevant documents are single-chunk. This is the
  long-document bias the design flagged.

## Title in reranker input (RRF + Ettin)

| Dataset      | Baseline | With title | Δ nDCG@10 |      p | Wins / ties / losses |
| ------------ | -------: | ---------: | --------: | -----: | -------------------- |
| SciFact      |  0.73252 |    0.74027 |   +0.0078 |   0.33 | 39 / 233 / 28        |
| NFCorpus     |  0.34269 |    0.34325 |   +0.0006 |   0.88 | 91 / 155 / 77        |
| ArguAna 300q |  0.60145 |    0.62991 |   +0.0285 | 0.0015 | 70 / 191 / 39        |

The title is prepended to 45% of SciFact and 48% of NFCorpus chunks (those that don't already
start with it). On those two datasets, the ones with real titles, the effect is small and not
significant.

**The ArguAna gain is not evidence that titles help.** ArguAna documents have no titles, and
the harness ingests untitled documents with their BEIR ID as the title (e.g.
`training-environment-pscisghwbs3-pro02b`). So on ArguAna the reranker was reading ID strings,
which carry the debate category and pro/con markers, prepended to 77% of chunks. Cross-posted
duplicate arguments account for only 3 of the 31 net wins; the rest plausibly come from the
category words in the IDs. That signal would not exist for real documents, whose titles are
file names or document titles.

## Caveats and follow-ups

- To test titles properly on argument-style data, re-prepare ArguAna with empty titles (or
  real debate titles) and compare again.
- ArguAna is a 300-query sample (seed 42); SciFact and NFCorpus used all test queries.
- Tuning covered five weight pairs and four fusion/order combinations at the chosen pair only.
