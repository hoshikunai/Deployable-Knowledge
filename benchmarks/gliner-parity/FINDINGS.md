# GLiNER ONNX findings (2026-10-05)

Question: can the entity relation pipeline run GLiNER inside the app (Node, `onnxruntime-node`,
no Python), and should it use GLiNER multitask or GLiNER-relex?

**Answer:**

- **Both can run in ONNX.** Relex is the better choice, exported as three graphs (`relex_split.py`).
  It matches PyTorch exactly, handles every input shape tried, and takes 0.85 s per chunk for
  entities and relations together.
- **Multitask** works through the community fp32 export at 1.58 s per chunk; its int8 file is
  broken.
- **Relex's single-graph export from `export_to_onnx()`** fails on every real input.

All runs: 22 fixed chunks (20 from the TCCC handbook, AFI 36-2406 and DAFMAN 36-2806 in `app.db`,
plus two general sentences), 8 entity labels, 4 relation types, CPU with 4 threads (the app's
`INFERENCE_THREADS` on this 16-core machine), `gliner` 0.2.29, `onnxruntime` 1.30.

## Multitask (`knowledgator/gliner-multitask-large-v0.5`)

| Variant                      | Matches PyTorch                                                              | Entity pass median | Relation pass median | s per chunk | Peak RAM |
| ---------------------------- | ---------------------------------------------------------------------------- | ------------------ | -------------------- | ----------- | -------- |
| PyTorch                      | reference                                                                    | 0.97 s             | 1.38 s               | 2.17        | 4.2 GB   |
| ONNX fp32 (`onnx-community`) | **yes**: 295/295 entities, 107/107 relations, logits within 2.6e-5           | 0.65 s             | 1.06 s               | **1.58**    | 4.1 GB   |
| ONNX int8 (`onnx-community`) | **no**: 40/295 entities, 7/107 relations, logits off by up to 9.4 (mean 3.0) | 0.28 s             | 0.55 s               | 0.74        | 2.3 GB   |

- **Output layout:** the fp32 graph returns `logits` as `[3, batch, words, classes]` (start, end,
  inside first), while `gliner` 0.2.29 expects classes last. `run.py` transposes it.
- **Extra outputs:** the graph also exposes four internal tensors, which can be ignored.
- **Relation pass:** asks `"<entity> <> <relation>"` for the 4 highest-scoring entities × 4
  relations (16 prompts in one pass). Its answers depend on which entities are picked as heads.

## Relex (`knowledgator/gliner-relex-large-v0.5`)

### Single graph (library export): unusable

| Check                                                      | Result                                                                                                                |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `export_to_onnx()` as shipped                              | fails: `aten::diag_embed` has no ONNX translation at opset 19 (GCN adjacency layer)                                   |
| With `diag_embed` replaced by `values.unsqueeze(-1) * eye` | exports a 1.87 GB graph                                                                                               |
| That graph on real inputs                                  | fails on 22 of 22 chunks (`ScatterND`: invalid index), on a text with no entities (`Reshape`) and on a two-text batch |

**Why it fails:**

- **Tracing freezes shapes.** The export uses TorchScript tracing. Entity selection packs kept spans
  to `lengths.max().item()` entities, which tracing turns into a constant from the example input.
- **Frozen pair building.** The pair builder's per-batch Python loop and its "no pairs" branch are
  frozen the same way.
- **Fixed thresholds.** The entity and adjacency thresholds are baked in at 0.5.

### Three graphs (`relex_split.py`): works

The data-dependent steps move out of ONNX into plain code, which is the code a TypeScript port
needs anyway.

| Graph            | Contents                                                                       | Size     |
| ---------------- | ------------------------------------------------------------------------------ | -------- |
| `encoder.onnx`   | Encoder, LSTM, span representations, entity scores, relation-prompt embeddings | 1,840 MB |
| `adjacency.onnx` | GCN adjacency over the selected entities (`diag_embed` replaced as above)      | 5 MB     |
| `pairs.onnx`     | Pair projection and relation scores against the relation prompts               | 28 MB    |

Glue between them, in numpy (about 60 lines):

1. Keep spans whose best class sigmoid is above the entity threshold, in flat span order.
2. Pack their representations and boundaries.
3. Pair entities off the diagonal where adjacency is above the adjacency threshold, in row-major
   order. Pad with entity 0, or return one masked pair when none qualify.

Only the two harmless tracer warnings remain (`_fit_length`, the context encoder's shape check), the
same ones the working multitask export has.

| Check                                                 | Result                                                                                                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Glue with the PyTorch layers vs PyTorch's own forward | identical on 23 of 23 cases (tensor difference 0)                                                                                                                |
| ONNX graphs vs PyTorch, tensors                       | entity logits within 2.4e-3, relation logits within 2e-5. One case of 23 had 518 instead of 520 candidate pairs: two adjacency scores within float noise of 0.5. |
| End to end through `model.inference()`                | **entities 235/235, relations 92/92**, scores identical to 4 decimals                                                                                            |
| Text with no entities, two-text batch                 | both work                                                                                                                                                        |
| Speed                                                 | **0.85 s per chunk** median for entities and relations together (PyTorch: 1.16 s)                                                                                |
| Encoder load (ONNX Runtime alone)                     | 32 s at basic optimization, 54 s at full, 24 s from a pre-optimized file. Save that file at the extended level: the full level writes hardware-specific code.    |
| Peak RAM                                              | Not measured for ONNX alone. The test harness also holds the PyTorch model for tokenizing and decoding (6.3 GB together).                                        |

That chunk with 520 candidate pairs shows an adjacency threshold of 0.5 is very loose. The relation
threshold (0.7) filters them, but a higher adjacency threshold (the fork used 0.55) would cut the
pair projection's work.

**Output quality, as an impression from three chunks** (not a measurement): relex gave more
sensible, direction-aware triples than multitask ("Splinting treats pelvic disruptions",
"Acetaminophen treats mild to moderate pain"), along with self-links and heading noise
("DEPARTMENT OF THE AIR FORCE responsible for Department of the Air Force").

## Porting to TypeScript

|                          | Relex (three graphs)                                                                                                                                                                                                                                                                                                                      | Multitask                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ONNX source              | `relex_split.py export` (needs the Python venv once; ship or download the three files)                                                                                                                                                                                                                                                    | `onnx-community` fp32, ready                                                                                             |
| Model work per chunk     | One encoder pass, plus two small graphs                                                                                                                                                                                                                                                                                                   | Entity pass, then a relation pass                                                                                        |
| Ported to TypeScript     | Processor with `<<ENT>> … <<SEP>> <<REL>> … <<SEP>>` prompt and span tensors (GLiNER.js `SpanModel` and `processor.ts`, Apache-2.0); the glue above; span decoder with greedy search (GLiNER.js `SpanDecoder`); relation decoder (about 130 lines in Python), mapping model entity indices to decoded spans through the entity boundaries | Processor, token tensors, token decoder (GLiNER.js `TokenModel` and `TokenDecoder`), relation prompts and their decoding |
| Thresholds               | Runtime parameters                                                                                                                                                                                                                                                                                                                        | Runtime parameters                                                                                                       |
| Speed (this CPU)         | 0.85 s per chunk                                                                                                                                                                                                                                                                                                                          | 1.58 s per chunk                                                                                                         |
| Verified against PyTorch | Yes, in Python                                                                                                                                                                                                                                                                                                                            | Yes, in Python                                                                                                           |

## Recommendation

Build on **relex with the three-graph export**. It's faster, extracts relations in the same pass
as entities, keeps thresholds adjustable, and is verified exactly against PyTorch.

Next steps for the port:

1. Write the TypeScript runtime (processor, the glue, decoders) on `onnxruntime-node`.
2. Check its output with the same parity method: TypeScript writes JSON and `compare` reads it
   against `relex-torch.json`.
3. Decide how the three ONNX files reach users: exported once and published to a model repository,
   or exported on the developer's machine and shipped.

The pipeline above the model (labels, cache, running state, resolution, dictionaries) doesn't change.
