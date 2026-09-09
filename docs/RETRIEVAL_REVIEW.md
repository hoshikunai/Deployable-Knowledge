Retrieval implementation review — 2026-09-08

Reviewed `feature/chunk-rating-v2` starting at `c339ba8`. The working tree was clean at review start; the earlier candidate-capture and benchmark work had already been committed. The starting revision, working-tree/index patches, and a consistent SQLite backup are preserved in `/tmp/dk-retrieval-review.8a8vgdun`. This review did not commit, stage, rebase, or change migration files.

The implementation is ready for further benchmark data collection, but the reconciled migration chain is **not ready to ship as an upgrade for existing experimental installations**. Retrieval improvement remains unproven.

Changes made during this review:

1. The fixed benchmark now uses the same expanded candidate limit as Search Context before learned reranking and truncates each ranking afterward. Previously, reranking only K candidates could change ordering but could never improve hybrid Recall@K by selecting a hidden candidate.
2. Benchmark creation validates that document filters exist and every judged chunk belongs to the selected documents, inside the creation transaction. Invalid cases leave no partial records.
3. A benchmark aborts if the effective learned model changes between cases, instead of labeling a mixed report with the last model ID.
4. Activation rejects missing/non-finite NDCG gate values and non-integer ranking-group counts.
5. Activation ranking metrics now count only hybrid groups, matching the mode in which learned ranking is deployed. Pairwise accuracy remains an aggregate diagnostic. The ranking strategy is now `pairwise-logistic-rank-blend-v3`, invalidating evaluation records produced under the previous rules. Dataset version remains 4 and feature version remains 2.

Verification completed:

- Fourteen integration test groups exercised the actual TypeScript domain modules and repositories against a disposable SQLite copy. They covered full candidate snapshots, hidden-result rating rejection, wrong-query/chunk attribution rejection, rating persistence across connections, human/proxy isolation and deletion, direct query-relative ranking, all 20 features and scaling, pair gaps and normalization, query-disjoint folds, current training persistence, activation gates, benchmark filtering, candidate expansion, and endpoint attribution. Retriever/model fixtures were substituted where necessary; these tests do not measure search quality.
- A human-source training run completed in the disposable database. Its input included two synthetic human-source fixture ratings created only for testing, so its metrics are not a clean evaluation of the user's feedback. It did not satisfy activation gates and did not activate a live model.
- The official Drizzle/libSQL migrator passed fresh replay and repeated replay. An upgrade from the consolidated schema before candidate capture preserved all 354 feedback IDs/links and 3,706 historical displayed result IDs/ranks copied from the backup. Foreign-key checks passed. Hidden rows were accepted and contradictory display states rejected.
- Drizzle generation against a copied migration directory reported no schema changes. No repository migrations were generated or edited.
- The built SvelteKit server returned successful workspace HTML, layout, heartbeat, active-model, benchmark-list, and empty-search responses with outbound inference-provider requests disabled. Benchmark/model/rating error paths returned the expected statuses. Browser hydration, mouse/keyboard star interactions, and a real semantic/cross-encoder search were not automated in this environment; no browser automation package or browser executable was available.
- `npm run lint`, `npm run check` (zero errors, zero warnings), `npm run build`, and `git diff --check` passed for the code changes. The build emitted the usual large-chunk warning and adapter-auto environment notice.

Outstanding migration findings:

- **Old experimental migration history cannot upgrade through the consolidated chain.** Replaying the original `d064b90` migrations succeeds; applying the current chain afterward fails with `table retrieval_feedback already exists`. The consolidated migration creates tables that those installations already have. The successful historical-row test above does not cover this distinct migration-ledger transition.
- **The current development database is not ready for desktop migration bootstrap.** Its migration ledger is empty although its tables already exist. Applying the official migrator to a disposable copy fails with `table api_keys already exists`. Development schema synchronization and packaged migration replay are different paths.
- Resolve these with a deliberate, versioned transition before shipping upgrades or moving this development database into the packaged application's data directory. Preserve the existing backups and validate both histories with real migrator replay. Do not mark migrations applied merely to suppress errors without establishing schema equivalence and preserving feedback/model links.

Evaluation limits to retain in the next project phase:

- Cross-validation learns with full recorded candidate context but evaluates the ordering of rated candidates only. Its rank blend is therefore evaluated on a smaller set than live inference. The independent benchmark remains necessary even after activation gates pass.
- Historical groups contain displayed results only. Deleted/re-ingested chunks can also change available candidate context and benchmark judgments. Freeze the document corpus, model configuration, top K, and judgments for baseline/learned comparisons.
- Training NDCG uses `2^rating - 1`; the fixed benchmark uses zero gain for unjudged/one-star chunks and `2^(rating - 1) - 1` for other ratings. These are different evaluation scales; do not compare their absolute NDCG values directly.
- Use `topK=20` for the planned benchmark. When K is below five, the current NDCG@5 result is computed from the available K results with missing positions contributing zero.
- The benchmark intentionally excludes direct exact-query human-feedback adjustments, isolating the learned ranker's contribution. It is not an end-to-end score for personalized Search Context results.

Evidence and runnable review harnesses are in `/tmp/dk-retrieval-review.8a8vgdun`: `review-tests.cjs`, `test-results.json`, `tests.log`, `migration-tests.cjs`, `migrations.log`, `legacy-migration.cjs`, `legacy-migration.log`, `current-db-migration.cjs`, `current-db-migration.log`, `server-smoke.mjs`, `server-smoke.log`, and gate logs. These are local review artifacts, not a permanent test framework; preserve them outside `/tmp` if longer retention is needed.
