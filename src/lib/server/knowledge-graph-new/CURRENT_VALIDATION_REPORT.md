# Knowledge Graph Current Validation Report

Date: August 4, 2026  
Branch: `kg-new`  
Reference: [Architecture and First-Run Findings](./ARCHITECTURE_AND_FINDINGS.md)

> **Historical result:** This report measured the `kg-v5` schema pipeline. The
> `kg-v20` adds bounded multi-pass discovery, deterministic relation/type closure
> and relation deduplication, dynamic extraction type enums, compact prompt
> labels, strict endpoint compatibility, generic role/group distinctions, and
> semantic-quality filters for templates, provenance, fragments, and citations.
> It also supplies exact GLiNER entity spans to the LLM as optional, untrusted
> hints while retaining conservative GLiNER relation thresholds, and requires
> endpoint mentions to occur at token boundaries rather than as substrings.
> Grounding checks run again during reconciliation so cached extractor output
> cannot bypass newer validation rules. Passive-use relations are rejected when
> their declared subject is a person, role, or person group. Paragraph, section,
> chapter, appendix, figure, and table locators remain provenance rather than
> document entities.
> Native extraction JSON schema uses one branch per canonical relation so a
> predicate cannot emit endpoint types outside its declared subject/object sets.
> Discovery requires relation names to read grammatically as subject-predicate-
> object and explicitly rejects passive-use names for actor subjects.
> Canonicalization merges relation names that differ only by leading auxiliary
> verbs, while publication “details procedure” edges remain provenance.
> None of the quality numbers below should be treated as measurements of
> `kg-v20`; rerun the complete benchmark before claiming improvement.

## Executive Summary

The current implementation does not meet the validation gates defined in
`ARCHITECTURE_AND_FINDINGS.md`.

Two improvements were confirmed:

- every retained assertion used a canonical relation from the discovered
  schema; and
- every retained assertion preserved an exact source evidence substring that
  contained both endpoints.

The remaining gates failed or could not be completed:

- the production schema request exceeded its effective context budget;
- schema discovery again returned only the ten universal entity types;
- three of five discovered relations referenced undeclared entity types;
- the verifier accepted fragments and document-layout relationships;
- modality was flattened into `asserted`;
- candidate-heavy verifier requests returned empty output after retries; and
- the 30-chunk benchmark could not be completed reliably.

The result is not ready for graph retrieval, PPR, multi-hop reasoning, or UI
integration. PPR would amplify whatever edges are present; it would not correct
incorrect predicates, endpoints, direction, modality, or document-layout
leakage.

## Scope and Environment

The validation used:

- 5,291 nonempty stored `TEXT` chunks in `app.db`;
- 11 source documents;
- a deterministic 30-chunk benchmark sample balanced across all 11 documents;
- a separate 48-chunk schema sample;
- Ollama model `gemma4:latest`;
- model size: 8B, Q4_K_M;
- CPU inference with no GPU offload;
- a 16,384-token Ollama context window;
- Python 3.14;
- GLiNER 0.2.27;
- CPU PyTorch 2.13.0; and
- PyArrow 25.0.0.

The run was read-only with respect to application data. No `kg_new_*` tables
were created, and no source chunks or repository code were modified by the
benchmark runner.

## Method

The test followed the recommended gates in the reference report:

1. Select 30 representative chunks across the available documents.
2. Discover a schema from the full stored corpus.
3. Run GLiNER across the selected 30 chunks at production thresholds.
4. Run LLM extraction per selected chunk.
5. Reconcile LLM and GLiNER candidates.
6. Run the LLM verifier on all schema-compatible candidates.
7. Preserve raw, intermediate, final, timing, and failure results in a
   checkpoint outside the repository.
8. Manually review every retained assertion against the source evidence and the
   report's quality definitions.

The exact process and complete runner are documented in
[Validation Reproduction Guide](./VALIDATION_REPRODUCTION_GUIDE.md).

## Production Schema Request Failure

The production defaults are:

| Setting                 |             Value |
| ----------------------- | ----------------: |
| Context window          |     16,384 tokens |
| Schema sample           |         48 chunks |
| Schema character limit  | 30,000 characters |
| Schema output allowance |      2,000 tokens |

The actual full request used 16,322 input tokens. That left approximately 62
tokens inside the 16,384-token context for a response allowed to use up to 2,000
tokens.

```text
prompt tokens + output allowance = required context
16,322       + 2,000            = 18,322

available context = 16,384
shortfall         =  1,938 tokens
```

All three production-default schema attempts returned an empty structured
response. No extraction benchmark can start successfully with these defaults.

## Diagnostic Schema Result

To continue the quality investigation without changing source code, the schema
sample character limit was overridden to 18,000 for a diagnostic run.

That request used 13,724 input tokens and completed after one retry in 517,406
milliseconds (8 minutes 37 seconds).

### Entity types

The schema contained only the universal types:

```text
person
organization
location
event
document
date
system
component
process
object
```

It discovered no reusable types for medicine, aviation, military units,
protocols, uniforms, publications, archives, evaluation criteria, ranks, or
regulatory provisions.

### Relation types

| Relation                   | Subject types        | Object types                   | Type-consistent |
| -------------------------- | -------------------- | ------------------------------ | --------------- |
| `is_component_of`          | object, component    | system, organization, document | Yes             |
| `governs_protocol`         | document             | protocol                       | No              |
| `requires_compliance_with` | event, process       | document                       | Yes             |
| `is_assigned_to`           | organization, person | military_unit, location        | No              |
| `defines_scope_for`        | protocol             | date, event                    | No              |

`protocol` and `military_unit` were referenced by relations but were absent from
the entity vocabulary. Three of the five relations were therefore not closed
over the discovered entity types.

## Extractor Results

GLiNER processed all 30 selected chunks in 79,633 milliseconds and produced 49
raw candidates. The LLM completed 11 chunks before the run was stopped. Nine of
those chunks reached a final verifier result.

| Measure                                              | Result |
| ---------------------------------------------------- | -----: |
| Selected chunks                                      |     30 |
| Selected documents                                   |     11 |
| LLM-completed chunks                                 |     11 |
| Fully finalized chunks                               |      9 |
| Finalized chunks with assertions                     |      5 |
| Raw LLM candidates in completed chunks               |      6 |
| Raw GLiNER candidates across all 30 chunks           |     49 |
| Raw GLiNER candidates in the 11 LLM-completed chunks |     36 |
| Retained assertions                                  |      5 |
| LLM-only retained assertions                         |      4 |
| GLiNER-only retained assertions                      |      0 |
| LLM and GLiNER agreement                             |      1 |

### Per-chunk results

|   # | Document and page         | LLM | GLiNER | Final outcome                               |
| --: | ------------------------- | --: | -----: | ------------------------------------------- |
|   1 | AFMAN 11-2T-38V2, page 6  |   0 |      0 | Empty                                       |
|   2 | AFMAN 11-2T-38V2, page 30 |   0 |      0 | Empty                                       |
|   3 | AFMAN 11-2T-38V2, page 52 |   1 |      3 | One LLM edge retained                       |
|   4 | TCCC handbook, page 22    |   1 |      4 | One LLM edge retained                       |
|   5 | TCCC handbook, page 64    |   1 |      5 | One agreement edge retained                 |
|   6 | TCCC handbook, page 114   |   2 |      0 | One LLM edge retained                       |
|   7 | BrownBook, page 6         |   0 |      0 | Empty                                       |
|   8 | BrownBook, page 13        |   0 |      0 | Empty                                       |
|   9 | BrownBook, page 20        |   0 |     11 | Verifier failed after three attempts        |
|  10 | DAFI 36-2903, page 25     |   1 |      0 | One LLM edge retained                       |
|  11 | DAFI 36-2903, page 86     |   0 |     13 | Repeated empty verifier output; interrupted |

## Retained Assertions and Manual Review

### 1. Failure sentence treated as an event

```text
Failed to request or did not abide by threat declarations
    --requires_compliance_with-->
threat declarations
```

- Source: LLM only
- Types: `event -> document`
- Stored status: `asserted`
- Review: reject
- Reasons: sentence fragment as an entity, questionable document typing, and
  lost failure/negative modality

GLiNER also proposed that `AWACS`, `AWACS/GCI`, and `GCI` were components of a
`tactical plan`. The verifier correctly rejected those three candidates but
retained the LLM fragment.

### 2. Medical equipment linked to a page header

```text
Ambu bag
    --is_component_of-->
TACTICAL COMBAT CASUALTY CARE HANDBOOK
```

- Source: LLM only
- Types: `object -> document`
- Stored status: `asserted`
- Review: reject
- Reasons: incorrect relationship inferred from a page header and inventory
  layout

GLiNER produced four variants connecting a student checklist to an instructor,
a student, or a sentence fragment. The verifier rejected those candidates but
accepted the layout edge.

### 3. Containment compatibility mapped to component membership

```text
WALK --is_component_of--> CASEVAC asset
```

- Source: LLM and GLiNER agreement
- Types: `object -> system`
- Stored status: `asserted`
- Review: reject as currently represented
- Reason: the evidence says WALK "fits into" a CASEVAC asset, which does not
  establish that it is a component of the asset

This was the strongest result in the sample, but the canonical predicate changes
the source meaning.

### 4. Appendix relationship retained as graph knowledge

```text
Appendix E
    --is_component_of-->
TACTICAL COMBAT CASUALTY CARE HANDBOOK
```

- Source: LLM only
- Types: `component -> document`
- Stored status: `asserted`
- Review: reject for the semantic graph
- Reason: valid document structure that belongs in provenance metadata

### 5. Citation heading treated as a protocol

```text
Title 10 USC Section 772
    --governs_protocol-->
When wearing by persons not on active duty authorized
```

- Source: LLM only
- Types: `document -> protocol`
- Stored status: `asserted`
- Review: reject
- Reasons: heading fragment as an endpoint and use of undeclared entity type
  `protocol`

## Candidate Fan-out and Verifier Failures

The BrownBook failure contained 11 overlapping GLiNER candidates made from
variants of:

```text
Professional Military Education Instructor
Professional Military Education Instructor and Curriculum Developer
Professional Military Education Instructors
```

and four academy names. The verifier returned an empty response after all three
attempts.

The DAFI 36-2903 failure contained 13 overlapping candidates involving:

```text
scarf
white net wind scarf
circle
clouds
darts
visor
```

and variants of `cap`, `service cap`, and `female service cap`. It entered the
same repeated empty-output pattern.

The current verifier sends every candidate, the complete source chunk, the
complete schema, and repeated evidence strings in one request. It does not
deduplicate aliases or nested spans and does not batch decisions.

## Quality Metrics

All five retained assertions were reviewed conservatively against the reference
report's definition of a useful, durable graph fact.

| Metric                      |       Result |      Required | Outcome |
| --------------------------- | -----------: | ------------: | ------- |
| Useful-triple precision     |   0 / 5 (0%) |        >= 85% | Fail    |
| Direction accuracy          |  3 / 5 (60%) |        >= 90% | Fail    |
| Endpoint-type accuracy      |  3 / 5 (60%) |        >= 90% | Fail    |
| Canonical-relation coverage | 5 / 5 (100%) |        >= 90% | Pass    |
| Exact evidence grounding    | 5 / 5 (100%) | 100% expected | Pass    |

If the WALK/CASEVAC assertion is judged useful despite the predicate mismatch,
useful precision becomes 20%, which still fails the 85% target.

Useful-assertion recall was not calculated because a complete manually authored
set of expected assertions for all selected chunks does not yet exist.

## Modality Results

All five retained assertions were stored as `asserted`:

```text
asserted:  5
negated:  0
uncertain: 0
```

The first assertion contains explicit failure and negative language but was
still flattened into `asserted`. The current three-value status does not
represent requirements, prohibitions, permissions, recommendations,
conditions, evaluation findings, or descriptive facts.

## Performance Results

| Stage                               |                             Result |
| ----------------------------------- | ---------------------------------: |
| Production-default schema           | Failed after three empty responses |
| Diagnostic schema                   |                         517,406 ms |
| GLiNER, 30 chunks                   |                          79,633 ms |
| LLM, 11 chunks including cold start |                 1,715,072 ms total |
| Warm LLM extraction                 |       133,904 ms average per chunk |
| Successful nontrivial verification  |                  83,155 ms average |

The historical report recorded 269 LLM chunks in approximately 22 minutes 52
seconds. The current model, hardware, context, prompts, structured-output mode,
and corpus differ, so this is not a controlled throughput comparison. It does
show that the current configuration is not practical for another full-corpus
run.

## Comparison With the Historical Findings

| Area                           | Historical result    | Current result                                             | Assessment                       |
| ------------------------------ | -------------------- | ---------------------------------------------------------- | -------------------------------- |
| Execution reliability          | 269/269, no failures | Default schema failed; candidate-heavy verification failed | Regressed                        |
| Corpus-specific entity types   | None                 | None                                                       | Unchanged                        |
| Canonical predicates           | 34.2%                | 100%                                                       | Improved                         |
| Mechanical grounding           | 100%                 | 100%                                                       | Preserved                        |
| GLiNER-only final contribution | 20.0%                | 0% in partial result                                       | Regressed or over-filtered       |
| Document-layout leakage        | Common               | Two of five retained assertions                            | Persists                         |
| Semantic errors                | Common               | Present in all retained assertions                         | Persists                         |
| Modality                       | Mostly asserted      | All asserted                                               | Persists                         |
| Manual usefulness              | Approximately 41.7%  | 0-20% in partial result                                    | Below both old result and target |

The historical run used a TCCC-focused corpus, while the current benchmark was
balanced across 11 heterogeneous documents. Exact percentages are therefore not
directly interchangeable. The validation targets are still applicable to both.

## Root Causes

### Token budgeting

The sampler budgets source characters, not the complete prompt tokens. It does
not reserve output tokens or a safety margin before calling the provider.

The complete prompt also contains avoidable repetition:

- 48 headers containing full 64-character document and chunk IDs;
- the complete `sampledChunkIds` list in every serialized schema;
- a full schema plus a second `allowedRelations` representation;
- the JSON output schema in the prompt and again in Ollama's native `format`;
- complete evidence repeated inside every verifier candidate; and
- the complete schema, including provenance-only fields, in each extraction and
  verifier request.

### Schema type consistency

Relations can reference arbitrary type strings. There is no post-discovery rule
requiring every relation endpoint type to exist in the entity vocabulary.

GLiNER receives only the declared entity types plus `other`, while LLM
extraction can emit any type string. Reconciliation then accepts `other` and
`unknown` as compatible with every relation, weakening the apparent type
constraints.

### Narrow-relation forcing

Closing the relation vocabulary solved uncontrolled predicate creation but
reduced the vocabulary to five relations. The model forced unsupported source
meanings into the nearest relation even though the prompt instructed it to omit
unrepresentable assertions.

### Schema quality gate

Implemented in extraction version `kg-v20`. Finalized schemas are now scored
before GLiNER or LLM extraction begins. The corpus-agnostic gate measures:

- endpoint-type closure;
- removal of provenance, co-occurrence, and vague relations;
- bounded endpoint-type sets;
- distinct relation families;
- retention of discovered relation concepts;
- representation across successful discovery batches; and
- consolidation-path reliability.

Closure and semantic-relation validity are hard requirements. The weighted
score must meet `KNOWLEDGE_GRAPH_SCHEMA_MIN_QUALITY_SCORE`, which defaults to
`0.8`. A failure stops extraction with per-metric diagnostics; a passing report,
including warnings, is stored under `schemaProvenance.qualityGate`. This gate is
structural and consolidation-focused—it does not replace the labeled assertion
and retrieval benchmark.

### Candidate duplication

GLiNER emits nested, abbreviated, expanded, and repeated endpoint variants. They
are merged only when normalized endpoints match closely enough after extraction.
They are not collapsed before verification.

### Document-structure filtering

Exact evidence checks validate text presence, not semantic value. There is no
pre-verifier rule for page headers, repeated document titles, appendices,
tables, inventory lists, section headings, or page-number artifacts.

### Modality representation

Implemented in extraction version `kg-v19`. Assertion status, modal force, and
conditional scope are now represented separately:

- `status` records whether the relationship is asserted, negated, or uncertain;
- `modality` records observed, habitual, required, recommended, permitted, or
  prohibited force;
- `modalityCue` preserves the exact words supporting that classification; and
- `condition` preserves an exact source substring limiting when the assertion
  applies.

Deterministic reconciliation rejects missing or non-verbatim cues, treats bare
`may` as permission only when the model and verifier support that reading,
requires possibility readings to use uncertain status, and prevents an asserted
prohibition from being flattened into a negated fact. The verifier independently
checks modality and condition. These fields are retained in graph provenance and
CSV exports.

The gold evaluator now measures modality separately from assertion status. A
fresh corpus benchmark is still required to measure the change's effect on
accepted-assertion precision and recall.

## Required Changes Before Retesting

1. Budget the complete prompt in tokens and reserve output plus a safety margin.
2. Remove full IDs and provenance-only schema fields from model prompts.
3. Validate that every relation endpoint type exists in the finalized entity
   vocabulary.
4. Dynamically constrain assertion types to that finalized vocabulary.
5. Stop treating `other` and `unknown` as universally schema-compatible.
6. Add a `no_matching_relation` mapping outcome or equivalent deterministic
   rejection path.
7. Deduplicate nested GLiNER spans and alias variants before verification.
8. Verify candidates in bounded batches and checkpoint each batch.
9. Move document structure to provenance before graph reconciliation.
10. Validate the implemented modality cue and conditional-scope representation
    on the labeled benchmark, including ambiguous `may`, prescriptive `will`,
    prohibitions, and conditional requirements.
11. Create complete expected-assertion labels for the benchmark so recall can be
    measured.
12. Repeat the full 30-50 chunk gate before running a complete corpus build.

## Repository Health

After the validation, all repository gates passed:

```text
npm run lint       passed
npm run check      passed with 0 errors and 0 warnings
npm run build      passed
git diff --check   passed
```
