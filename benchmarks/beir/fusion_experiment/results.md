# Fusion experiment results

RRF weights, fusion level, document ordering and title input. 100→40→20, 1,200-character chunks, nomic-embed-text v1.5. Tuned on SciFact train and NFCorpus dev; reported on test.

## Tuning (RRF only)

| Configuration | nfcorpus dev nDCG@10 | scifact train nDCG@10 | Average |
|---|---|---|---|
| weights 1:0.5 · chunk fusion · order none | 0.33980 | 0.74125 | 0.54052 |
| weights 1:0.5 · document fusion · order none | 0.33862 | 0.73924 | 0.53893 |
| weights 1:0.75 · chunk fusion · order none | 0.33913 | 0.73363 | 0.53638 |
| weights 1:1 · chunk fusion · order none | 0.33635 | 0.73360 | 0.53497 |
| weights 0.75:1 · chunk fusion · order none | 0.33453 | 0.73302 | 0.53377 |
| weights 0.5:1 · chunk fusion · order none | 0.33093 | 0.72798 | 0.52946 |
| weights 1:0.5 · chunk fusion · order sum-top2 | 0.30467 | 0.59216 | 0.44842 |
| weights 1:0.5 · document fusion · order sum-top2 | 0.27845 | 0.52155 | 0.40000 |

Chosen: `{"rrfSemanticWeight": 1.0, "rrfBm25Weight": 0.5, "fusionLevel": "chunk", "documentOrder": "none", "rerankTitle": "off"}` (average nDCG@10 0.54052).

## scifact test (300 queries)

| Variant | nDCG@10 | Recall@10 | MRR@10 | Δ nDCG vs baseline | p | Wins/ties/losses | Chunks per top-10 doc | Found by both |
|---|---|---|---|---|---|---|---|---|
| RRF · baseline | 0.73377 | 0.86411 | 0.69790 | – | – | – | 1.96 | 33.1% |
| RRF · tuned | 0.73214 | 0.85689 | 0.70110 | -0.00163 | 0.8178 | 33/243/24 | 1.96 | 33.1% |
| RRF + Ettin · baseline | 0.73252 | 0.85078 | 0.69947 | – | – | – | 2.08 | 28.4% |
| RRF + Ettin · title | 0.74027 | 0.85744 | 0.70937 | +0.00775 | 0.3325 | 39/233/28 | 2.03 | 28.9% |
| RRF + Ettin · tuned | 0.72637 | 0.84233 | 0.69375 | -0.00615 | 0.4653 | 35/231/34 | 2.11 | 28.6% |
| RRF + Ettin · tuned + title | 0.73438 | 0.84456 | 0.70487 | +0.00186 | 0.8469 | 42/230/28 | 2.06 | 29.5% |

## nfcorpus test (323 queries)

| Variant | nDCG@10 | Recall@10 | MRR@10 | Δ nDCG vs baseline | p | Wins/ties/losses | Chunks per top-10 doc | Found by both |
|---|---|---|---|---|---|---|---|---|
| RRF · baseline | 0.35447 | 0.17221 | 0.56386 | – | – | – | 1.99 | 29.9% |
| RRF · tuned | 0.35396 | 0.16591 | 0.57087 | -0.00052 | 0.8528 | 68/190/65 | 1.97 | 29.8% |
| RRF + Ettin · baseline | 0.34269 | 0.16490 | 0.54888 | – | – | – | 2.05 | 23.8% |
| RRF + Ettin · title | 0.34325 | 0.16423 | 0.55361 | +0.00055 | 0.8794 | 91/155/77 | 2.01 | 24.9% |
| RRF + Ettin · tuned | 0.33654 | 0.16342 | 0.54167 | -0.00615 | 0.1099 | 76/167/80 | 2.05 | 24.4% |
| RRF + Ettin · tuned + title | 0.33810 | 0.16171 | 0.55110 | -0.00460 | 0.2751 | 88/139/96 | 2.01 | 24.7% |

## arguana test (300 queries)

| Variant | nDCG@10 | Recall@10 | MRR@10 | Δ nDCG vs baseline | p | Wins/ties/losses | Chunks per top-10 doc | Found by both |
|---|---|---|---|---|---|---|---|---|
| RRF · baseline | 0.54949 | 0.83000 | 0.46075 | – | – | – | 1.43 | 55.8% |
| RRF · tuned | 0.53809 | 0.83000 | 0.44661 | -0.01141 | 0.0997 | 33/203/64 | 1.43 | 56.0% |
| RRF + Ettin · baseline | 0.60145 | 0.83000 | 0.52725 | – | – | – | 1.39 | 43.5% |
| RRF + Ettin · title | 0.62991 | 0.86667 | 0.55296 | +0.02846 | 0.0015 | 70/191/39 | 1.38 | 44.0% |
| RRF + Ettin · tuned | 0.60460 | 0.85000 | 0.52551 | +0.00315 | 0.6293 | 37/227/36 | 1.40 | 43.4% |
| RRF + Ettin · tuned + title | 0.62689 | 0.87667 | 0.54648 | +0.02543 | 0.0053 | 71/182/47 | 1.39 | 44.1% |

