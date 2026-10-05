# Relex extraction quality on 2WikiMultiHopQA

Measures how well the app's relation model (`src/lib/server/entities/relex`) extracts relations,
by comparing its triples with the gold evidence triples 2WikiMultiHopQA provides for every dev
question. It runs the app's own TypeScript runtime, so the numbers describe what the app will do.

Each sampled question contributes its gold paragraphs (the ones its supporting facts cite) and
its gold `(subject, relation, object)` triples. Relex reads each paragraph separately, as it will
read document chunks, with the 2Wiki relation names as its relation labels.

## Before you start

- The relation model must be in `models/gliner-relex/`
  (`benchmarks/gliner-parity/README.md` describes the export).
- Python 3 with the standard library is enough for the build and scoring scripts.

## Run

From the repository root:

```bash
mkdir -p .cache/relex-eval/source
curl -L -o .cache/relex-eval/source/data_ids_april7.zip \
  "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
unzip -o .cache/relex-eval/source/data_ids_april7.zip -d .cache/relex-eval/source
DEV=$(find .cache/relex-eval/source -name dev.json | head -1)

python3 benchmarks/relex-eval/build_set.py --source "$DEV"
node_modules/.bin/tsx benchmarks/relex-eval/run_ts.mts
python3 benchmarks/relex-eval/score.py
```

The download is the dataset version with fixed sentence segmentation, linked from
<https://github.com/Alab-NII/2wikimultihop> (Apache-2.0).

- **`build_set.py`** samples 500 questions (`--count`, `--seed 42`), stratified by question type, and
  writes `.cache/relex-eval/2wiki-set.json`.
  - It prints the relation names, which become relex's relation labels, and how many gold triples
    are findable: both entities appear word for word in the paragraphs.
  - Gold triples come from Wikidata, so some use wording the text doesn't (a full date where the
    text has a year, an alternative name).
  - The entity labels default to `person, film, organization, location, country, date, award, other`
    (`--entity-labels`). `other` lets relex keep an entity whose type isn't listed when a relation
    needs it.
- **`run_ts.mts`** runs relex on every gold paragraph and appends one JSON line per question to
  `.cache/relex-eval/2wiki-ts.jsonl`.
  - It keeps every relation scoring above 0.3 (`--relation-floor`), so the scoring can try several
    thresholds from one run.
  - Entity and adjacency thresholds are the app defaults.
  - It resumes where it stopped if interrupted.
  - `--limit 20` makes a quick trial run.
  - Expect roughly 1,250 paragraphs at under a second each.
- **`score.py`** writes `.cache/relex-eval/2wiki-results.md` and `2wiki-summary.json`.

## Reading the results

- **Recall:** the share of gold triples relex found with the same relation and both entities
  matching.
  - _Exact_ matching compares normalized names (case, punctuation and articles removed).
  - _Lenient_ also accepts one name containing the other, or a token overlap of at least 0.5, so
    "Lyon" matches "Lyon, France".
- **Recall (findable):** the same, over gold triples whose entities are written in the text. This
  is the fairest measure of the model itself.
- **Precision vs gold:** the share of relex's triples that match a gold triple. It's a lower bound:
  the paragraphs state many true facts the gold set leaves out.
- **Reversed:** gold triples found only with subject and object swapped.
- **Why findable gold triples were missed:** each miss at threshold 0.7 sorted into one cause:
  - an entity wasn't extracted at all;
  - both entities were found but no matching relation;
  - the relation was found but scored below the threshold.
- **By relation / by question type:** recall split by 2Wiki relation and by question type.

## Tests

```bash
python3 -m unittest discover -s benchmarks/relex-eval
```
