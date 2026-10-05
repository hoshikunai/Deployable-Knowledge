# GLiNER ONNX parity

Checks whether the community ONNX export of GLiNER multitask
(`onnx-community/gliner-multitask-large-v0.5`) matches the original PyTorch model
(`knowledgator/gliner-multitask-large-v0.5`), and how fast each runs on this machine's CPU.
Everything here is Python; no app code is involved yet.

## Setup

```bash
uv venv --python 3.12 benchmarks/gliner-parity/.venv
uv pip install --python benchmarks/gliner-parity/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  torch "gliner[onnx]" onnx psutil
```

The models download into the Hugging Face cache on first use: about 1.8 GB for the PyTorch
weights and 2.4 GB for the fp32 and int8 ONNX files.

## Run

```bash
PY=benchmarks/gliner-parity/.venv/bin/python
python3 benchmarks/gliner-parity/make_texts.py
$PY benchmarks/gliner-parity/run.py torch
$PY benchmarks/gliner-parity/run.py onnx-fp32
$PY benchmarks/gliner-parity/run.py onnx-int8
$PY benchmarks/gliner-parity/compare.py
```

`make_texts.py` samples 20 prose chunks from `app.db` (the TCCC handbook and two Air Force
instructions) plus two general sentences, and fixes the entity labels and relation types.
Outputs go to `.cache/gliner-parity/`, which is not committed because it contains document text.

Each variant runs in its own process so its peak memory is measured alone. The torch run goes
first: it writes the relation prompts (`"<entity> <> <relation>"` for its four highest-scoring
entities per chunk), and the ONNX runs reuse them so all three answer the same questions.
Threads match the app's `INFERENCE_THREADS` (a quarter of the cores, at most 8).

## What is compared

| Check                | Meaning                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| Tokens match         | Both repositories' tokenizers produce identical input IDs                      |
| Max logit diff       | Largest absolute difference in raw model output on identical inputs            |
| Max prob diff        | The same after the sigmoid, which is what thresholds see                       |
| Entity / relation F1 | Agreement of the found (start, end, label) spans with PyTorch at threshold 0.5 |
| Max score diff       | Largest score difference on spans both found                                   |

`compare.py` exits non-zero unless fp32 matches exactly (F1 1.0, scores within 0.01) and int8
reaches F1 0.95 for both entities and relations. Timing covers the entity pass, the relation pass
and the seconds per chunk for both together.

## Relex

```bash
$PY benchmarks/gliner-parity/relex_export.py
$PY benchmarks/gliner-parity/relex_parity.py torch
$PY benchmarks/gliner-parity/relex_parity.py onnx
$PY benchmarks/gliner-parity/relex_parity.py compare
```

`relex_export.py` exports `knowledgator/gliner-relex-large-v0.5` with the library's
`export_to_onnx()` into `.cache/gliner-parity/relex-onnx/`, replacing `torch.diag_embed` (which has
no ONNX translation) with an equivalent product. `relex_parity.py` runs PyTorch and the export over
the same chunks at entity, adjacency and relation thresholds 0.5, 0.5 and 0.7, plus two edge cases:
a text with no entities and a two-text batch. Each failure is recorded per chunk rather than
stopping the run. That single-graph export fails on every input.

The working route exports relex as three graphs and does entity selection and pairing in numpy:

```bash
$PY benchmarks/gliner-parity/relex_split.py export
$PY benchmarks/gliner-parity/relex_split.py tensors
$PY benchmarks/gliner-parity/relex_parity.py split
$PY benchmarks/gliner-parity/relex_parity.py compare --candidate split
```

`relex_split.py export` writes `encoder.onnx`, `adjacency.onnx` and `pairs.onnx` to
`.cache/gliner-parity/relex-split/`. `tensors` compares the raw outputs (entity logits, selected
entities, pairs, relation logits) with PyTorch's forward pass, once with the glue driving the
PyTorch layers and once with the ONNX graphs. `relex_parity.py split` runs `model.inference()` with
its forward pass routed through the three graphs, so tokenizing and decoding stay the library's own.

Results and the TypeScript port assessment are in [FINDINGS.md](FINDINGS.md).

## The app's TypeScript runtime

The app runs relex through `src/lib/server/entities/relex/`, reading the graphs from
`models/gliner-relex/`. Export them there, then check the TypeScript runtime against PyTorch:

```bash
$PY benchmarks/gliner-parity/relex_split.py export --out models/gliner-relex
node_modules/.bin/tsx benchmarks/gliner-parity/relex_ts.mts
$PY benchmarks/gliner-parity/relex_parity.py compare --candidate ts
```

`relex_ts.mts` writes `.cache/gliner-parity/relex-ts.json` in the same shape as the Python runs.
