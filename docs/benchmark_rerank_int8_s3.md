# Reranking benchmark (session 3)

Date: 2026-11-28 00:30 | gold set: `eval/gold/retrieval_s2.csv` (15 questions) | first stage: dense / hybrid (dense + BM25 language-aware + BGE-M3 sparse, RRF k=60) | CPU

| Pipeline | Recall@5 | MRR@10 | exact MRR | paraphrase MRR | Rerank mean (ms) | Rerank p95 (ms) |
|---|---|---|---|---|---|---|
| dense | 93% | 0.83 | 0.90 | 0.80 | - | - |
| hybrid | 87% | 0.84 | 1.00 | 0.76 | - | - |
| dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1 int8, 10] | 93% | 0.90 | 1.00 | 0.85 | 373 | 433 |
| hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1 int8, 10] | 93% | 0.90 | 1.00 | 0.85 | 401 | 461 |
| dense -> rerank[bge-reranker-v2-m3 int8, 10] | 100% | 0.97 | 1.00 | 0.95 | 7402 | 8373 |
| hybrid -> rerank[bge-reranker-v2-m3 int8, 10] | 93% | 0.93 | 1.00 | 0.90 | 8123 | 9001 |

## Models

| Model | Parameters | Load time (s) |
|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | 5.6 |
| `BAAI/bge-reranker-v2-m3` | 568M | 5.1 |

## Latency budget of the best pipeline (mean per question, ms)

| Stage | Measured (ms) | SLA (ms) | Status |
|---|---|---|---|
| 1. Query encoding (BGE-M3) | 352.2 | < 30 | over SLA |
| 2. Dense search Qdrant (HNSW) | 10.8 | < 40 | OK |
| 3. Sparse search BM25 (in memory) | 2.3 | < 15 | OK |
| 3b. Sparse search BGE-M3 (Qdrant) | 9.9 | < 15 | OK |
| 4. RRF fusion | 0.1 | < 5 | OK |
| 5. Cross-encoder reranking (bge-reranker-v2-m3 int8, 10) | 7401.6 | < 150 | over SLA |

## Before / after reranking (evaluation sheet, section 2)

Question: "What is the withdrawal period after the delivery of equipment bought remotely with an Orange subscription?"

| Rank | Dense search only | Cosine | After reranking (best pipeline) | Rank before | Cross-encoder |
|:-:|---|:-:|---|:-:|:-:|
| 1 | Conditions générales de vente Mobicarte / Préambule (suite 3) | 0.7193 | (CGU) Option Sauvegarde en ligne / Article 12. Vente à distance | 6 | 0.507 |
| 2 | (CS) Changement de mobile / Article 4. Droit de rétractation en cas... | 0.7192 | (CGA) Services Internet et Mobiles / Article 5. Description du service | 7 | 0.506 |
| 3 | (CGU) Option Illimité ligne Livebox vers mobiles / Article 9. Vente... | 0.7049 | Conditions générales de vente Mobicarte / Préambule (suite 3) | 1 | 0.498 |
| 4 | (CG) Prestations d’assistance à distance / Article 10. DROIT DE RÉT... | 0.6990 | (CS) Changement de mobile / Article 4. Droit de rétractation en cas... | 2 | 0.449 |
| 5 | (CS) Prestations démarrage Smartphone pour les clients Origami Jet ... | 0.6983 | (CG) Prestations d’assistance à distance / Article 10. DROIT DE RÉT... | 4 | 0.250 |

## Rank of the first correct result per question (- = not in the top 10)

| # | Question | dense | hybrid | dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1 int8, 10] | hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1 int8, 10] | dense -> rerank[bge-reranker-v2-m3 int8, 10] | hybrid -> rerank[bge-reranker-v2-m3 int8, 10] |
|---|---|---|---|---|---|---|---|
| 1 | Quel est le prix mensuel de l'offre 4G Home 200 Go ? | 1 | 2 | 1 | 1 | 1 | 1 |
| 2 | Combien coûte le forfait 500 Go 5G+ pour un client q... | 1 | 1 | 1 | 1 | 1 | 1 |
| 3 | Le prix de la location de la Livebox est-il inclus d... | 2 | 1 | 1 | 1 | 1 | 1 |
| 4 | Le forfait 500 Go 5G+ est-il disponible sans engagem... | 3 | 1 | 1 | 1 | 1 | 1 |
| 5 | Y a-t-il des frais de résiliation pour la 4G Home ? | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | Quelle est la durée d'engagement de l'offre Série Sp... | 1 | 9 | 1 | 1 | 1 | 1 |
| 7 | Comment résilier l'option internet tranquillité 50 Mo ? | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | What is the withdrawal period after the delivery of ... | 7 | - | 3 | - | 2 | - |
| 9 | How many days do I have to withdraw after ordering a... | 1 | 1 | 6 | 1 | 1 | 1 |
| 10 | What is the maximum 4G+ download speed on the 500 GB... | 1 | 1 | 1 | 2 | 1 | 1 |
| 11 | Le prix de la location Livebox est-il inclus dans l'... | 1 | 1 | 1 | 1 | 1 | 1 |
| 12 | Quel est le montant des frais de construction pour u... | 1 | 1 | 1 | 1 | 1 | 1 |
| 13 | Quel est le prix de la Mobicarte Start 150Go ? | 1 | 1 | 1 | 1 | 1 | 1 |
| 14 | Quel est le délai de rétractation de l'option Family... | 2 | 1 | 1 | 1 | 1 | 1 |
| 15 | Quel est le prix mensuel de l'offre Mon Optimale Int... | 1 | 1 | 1 | 1 | 1 | 1 |

## Reading

Experiment: int8 dynamic quantization of the Linear layers (`--int8`), 10 candidates, compared with the fp32 runs of `benchmark_rerank_s3.md`.

- mmarco-mMiniLMv2: 0.71 s -> 0.37 s (1.9x faster), MRR 0.89 -> 0.90, but Recall@5 100% -> 93%: one question leaves the top 5 that is sent to the LLM.
- bge-reranker-v2-m3: 15.1 s -> 7.4 s (2x faster), MRR 1.00 -> 0.97: one question is no longer at rank 1.

**Decision:** not used (`RERANKER_INT8=0`). The speed-up does not reach the 150 ms SLA and costs one question on a 15-question gold set.
