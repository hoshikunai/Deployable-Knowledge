# New Test Results running pdf-eval:

## DK Chunking Results:

### pdf-eval · App default (union + Ettin 32M) · 40→20 · no title in reranker input · 1,200-character chunks · nomic-embed-text v1.5 (768-d)
`pdf-eval-default-p40-o20-all-20261116-161649-134275` · complete · 81/81 queries (all, test split)

#### Retrieval metrics @10

| Method | nDCG | Recall | MRR | MAP | P |
|---|---|---|---|---|---|
| bm25 | 0.71085 | 0.85288 | 0.69239 | 0.64630 | 0.10494 |
| semantic | 0.66860 | 0.78704 | 0.66008 | 0.60814 | 0.10494 |
| hybrid | 0.79023 | 0.87449 | 0.79636 | 0.74154 | 0.11358 |

_No chunk scores recorded (run predates score recording)._


## Docling Chunking Results:

### pdf-eval · App default (union + Ettin 32M) · 40→20 · no title in reranker input · 1,200-character chunks · nomic-embed-text v1.5 (768-d) · Docling + TableFormer
`pdf-eval-default-p40-o20-docling-all-20261116-162235-429832` · complete · 81/81 queries (all, test split)

#### Retrieval metrics @10

| Method | nDCG | Recall | MRR | MAP | P |
|---|---|---|---|---|---|
| bm25 | 0.71124 | 0.85288 | 0.69294 | 0.64665 | 0.10494 |
| semantic | 0.68789 | 0.80864 | 0.67963 | 0.62602 | 0.10617 |
| hybrid | 0.82300 | 0.90947 | 0.82551 | 0.77529 | 0.11852 |

_No chunk scores recorded (run predates score recording)._

