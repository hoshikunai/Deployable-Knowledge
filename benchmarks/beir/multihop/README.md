# Multi-hop pooled corpora

The single-hop BEIR corpora (SciFact, NFCorpus, ArguAna, …) ask questions one document can
answer. These corpora ask questions that need two to four documents. Full multi-hop corpora
are too large for a laptop (BEIR HotpotQA has 5.2M passages), so `build_pool.py` builds a
**pooled** corpus instead: a sample of questions, their gold paragraphs, and the hard
distractor paragraphs each dataset ships with, in the BEIR layout the harness already reads.

Scores on a pool are only comparable with other runs on the same pool, not with published
leaderboards.

## Single-hop vs multi-hop

Every multi-hop dataset name starts with `multihop-`, and its catalog entry in
`dataset_catalog.py` has `kind=MULTI_HOP`. Run IDs (`<dataset>-<run name>`) and corpus
copies (`.cache/beir/runtime-<dataset>-<variant>`) therefore show the kind too. The builder
refuses names without the prefix.

| Dataset                    | Source                                   | Paragraphs per question         | Types                                                    |
| -------------------------- | ---------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| `multihop-hotpotqa`        | HotpotQA distractor dev (CC BY-SA 4.0)   | 2 gold + 8 TF-IDF distractors   | bridge, comparison                                       |
| `multihop-musique`         | MuSiQue-Ans dev (CC BY 4.0)              | 2–4 gold + the rest of 20       | 2-hop, 3-hop, 4-hop                                      |
| `multihop-2wiki`           | 2WikiMultiHopQA dev (Apache-2.0)         | 2–4 gold + the rest of 10       | comparison, inference, compositional, bridge_comparison  |
| `multihop-hotpotqa-filler` | `multihop-hotpotqa` + HotpotQA train paragraphs | same questions, larger corpus | as `multihop-hotpotqa`                             |

## Get the source files

Put the raw files under `benchmarks/beir/datasets/_sources/` (ignored by Git):

| Dataset  | File                                                          | Where                                              |
| -------- | ------------------------------------------------------------- | -------------------------------------------------- |
| HotpotQA | `hotpot_dev_distractor_v1.json`, `hotpot_train_v1.1.json`     | https://hotpotqa.github.io (download links)        |
| MuSiQue  | `musique_ans_v1.0_dev.jsonl`                                  | https://github.com/StonyBrookNLP/musique (`download_data.sh`) |
| 2Wiki    | `dev.json`                                                    | https://github.com/Alab-NII/2wikimultihop (dataset link in README) |

## Build

From the repository root, while no benchmark is running (the HotpotQA train file needs several
GB of memory to load):

```bash
PY=benchmarks/beir/.venv/bin/python
SRC=benchmarks/beir/datasets/_sources

$PY benchmarks/beir/multihop/build_pool.py --dataset hotpotqa \
  --source $SRC/hotpot_dev_distractor_v1.json --name multihop-hotpotqa
$PY benchmarks/beir/multihop/build_pool.py --dataset musique \
  --source $SRC/musique_ans_v1.0_dev.jsonl --name multihop-musique
$PY benchmarks/beir/multihop/build_pool.py --dataset 2wiki \
  --source $SRC/dev.json --name multihop-2wiki
$PY benchmarks/beir/multihop/build_pool.py --dataset hotpotqa \
  --source $SRC/hotpot_dev_distractor_v1.json --name multihop-hotpotqa-filler \
  --filler 40000 --filler-source $SRC/hotpot_train_v1.1.json
```

Defaults: `--per-split 1000` (a `dev` split for tuning and a disjoint `test` split for
reporting), `--seed 42`. The filler build uses the same source and seed, so it samples the
same questions as `multihop-hotpotqa`. An existing dataset directory is only replaced with
`--force`; rebuilding invalidates its prepared corpus copies, so prepare them again afterwards.

Then prepare each corpus in `npm run bench` (Settings → corpus copies, or Prepare on the Run tab).

## What a pool contains

| File                    | Contents                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `corpus.jsonl`          | Gold and distractor paragraphs of every sampled question, deduplicated on (title, text)          |
| `queries.jsonl`         | The sampled questions                                                                            |
| `qrels/dev.tsv`, `qrels/test.tsv` | Gold paragraphs per question, score 1                                                  |
| `multihop.json`         | Per question: split, type, level, answers, supporting facts, MuSiQue decomposition, gold hops    |
| `manifest.json`         | Seed, source SHA-256, sampled IDs, drop counts, corpus counts, hop-role coverage                  |
| `evidence-triples.json` | 2Wiki only: gold (subject, relation, object) triples per question                               |

Questions are dropped before sampling when a gold paragraph has fewer than five words (the
harness cannot index it), when they have fewer than two gold paragraphs, when a context repeats
a title, or (MuSiQue) when they are unanswerable. Distractors under five words are left out of
the corpus. Malformed source records stop the build.

**Hop roles.** MuSiQue's come from its question decomposition. For HotpotQA and 2Wiki a gold
paragraph whose title appears in the question is hop 1 and the others hop 2; when no title
appears, the hops stay unknown. `manifest.json` counts both.

**Filler** paragraphs come from the train file, skipping titles already in the pool or named in a
sampled question.

## Metrics

The Scores page adds a multi-hop section for these corpora, computed from saved rankings:

- **Complete@k**: share of questions with *every* gold paragraph in the top k. BEIR's Recall@10
  gives 0.5 when one of two hops is found, which hides the failure that matters here.
- **By type**: Complete@10 and Recall@10 per question type (or hop count for MuSiQue). HotpotQA
  comparison questions name both entities, so they behave almost like single-hop questions.
- **By hop**: Recall@10 for hop-1 and hop-2+ gold paragraphs. A low hop-2+ recall means the second
  document is not reachable from the question text.
- Paired comparisons add a sign-flip test on per-question Complete@10.

Expect semantic scores to be inflated on HotpotQA: nomic and bge were trained on HotpotQA train
data. MuSiQue is the less contaminated check.

## Tests

```bash
benchmarks/beir/.venv/bin/python -m unittest discover -s benchmarks/beir/multihop
```
