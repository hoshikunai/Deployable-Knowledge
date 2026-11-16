# New Test Results running pdf-eval:

## DK Chunking Results:

### pdf-eval · App default (union + Ettin 32M) · 40→20 · no title in reranker input · 1,200-character chunks · nomic-embed-text v1.5 (768-d)
`pdf-eval-default-p40-o20-all-20261116-180551-878808` · complete · 81/81 queries (all, test split)

#### Retrieval metrics @10

| Method | nDCG | Recall | MRR | MAP | P |
|---|---|---|---|---|---|
| bm25 | 0.71646 | 0.85288 | 0.70773 | 0.65214 | 0.10494 |
| semantic | 0.64633 | 0.80556 | 0.61298 | 0.57715 | 0.10741 |
| hybrid | 0.78714 | 0.90535 | 0.78098 | 0.72741 | 0.11852 |

_No chunk scores recorded (run predates score recording)._



## Docling Chunking Results:

### pdf-eval · App default (union + Ettin 32M) · 40→20 · no title in reranker input · 1,200-character chunks · nomic-embed-text v1.5 (768-d) · Docling + TableFormer
`pdf-eval-default-p40-o20-docling-all-20261116-180013-146819` · complete · 81/81 queries (all, test split)

#### Retrieval metrics @10

| Method | nDCG | Recall | MRR | MAP | P |
|---|---|---|---|---|---|
| bm25 | 0.72403 | 0.85288 | 0.71761 | 0.66202 | 0.10494 |
| semantic | 0.67139 | 0.82716 | 0.63973 | 0.60220 | 0.10864 |
| hybrid | 0.79706 | 0.91152 | 0.79707 | 0.73796 | 0.11852 |

_No chunk scores recorded (run predates score recording)._


