Fixed benchmark preparation — 2026-09-08

Steps 3 and 4 have begun. Twelve broad-corpus queries and their baseline rankings are captured. Independent human judgments are still required before cases can be finalized and relevance metrics reported.

The local working packet is `.cache/retrieval-benchmarks/20260908-broad-corpus/README.md`. It links to 12 shuffled judging packets and a single `judgments.csv` with 416 blank rating cells. The candidate union contains the top 20 results from semantic, BM25, and hybrid baseline retrieval for each query. All 11 active corpus documents are represented. No ratings were inferred or copied from training feedback.

The run used the actual cached embedding and cross-encoder models over a consistent copy of the 5,370-chunk corpus. The learned model was inactive and direct exact-query feedback adjustments were excluded. All four ranking outputs returned 20 unique chunks per query; hybrid learned matched hybrid baseline because no learned model was active. The 12 searches took approximately 117 seconds in total, including model initialization.

The corpus SHA-256 is `1634f0d7a612b8ba4c076ddc25ac10397cde2d9fcc193da49fabc31e0979862b`. Metadata also records model IDs, scoring version, dependency-lock hash, working-diff hash, runtime bundle hash, query-set hash, and capture times. The queries were drafted by the assistant from the available corpus and checked to avoid exact normalized matches to existing human/proxy feedback queries. That check does not imply the query topics are unrelated to prior training topics.

Required next actions:

1. Have a human judge the packets with the agreed 1–5 labels and fill only the `relevance` column in `judgments.csv`. Each case must have at least one independently judged 4- or 5-star chunk. Do not create training ratings for these held-out query strings.
2. Run `python3 .cache/retrieval-benchmarks/20260908-broad-corpus/check-judgments.py`. The validator refuses incomplete sheets, malformed values, duplicate/unknown IDs, and cases with no relevant chunk. `--write-payloads` exports requests after validation and performs no database writes.
3. Recheck corpus/model configuration stability, import the completed cases, and record the fixed baseline metrics with the learned model inactive. The current captures are unscored rankings; no Recall@20, reciprocal-rank, or NDCG@5 result is claimed yet.
4. Preserve the same corpus, query set, and judgments for subsequent comparison. If the learned model retrieves previously unjudged chunks, judge those independently and recompute both baseline and learned metrics on the same expanded judgment set.

All retrieval in this preparation used the copied database. Live human/proxy feedback, retrieval impressions, saved models, active-model state, and benchmark cases were left unchanged. The migration blockers described in `RETRIEVAL_REVIEW.md` remain outside this benchmark preparation work. The judging packet and snapshot are Git-ignored local data, not committed corpus content.
