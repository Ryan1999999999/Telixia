# Reranking benchmark (session 3)

Date: 2026-11-27 20:46 | gold set: `eval/gold/retrieval_s2.csv` (15 questions) | first stage: dense / hybrid (dense + BM25 language-aware + BGE-M3 sparse, RRF k=60) | CPU

| Pipeline | Recall@5 | MRR@10 | exact MRR | paraphrase MRR | Rerank mean (ms) | Rerank p95 (ms) |
|---|---|---|---|---|---|---|
| dense | 93% | 0.83 | 0.90 | 0.80 | - | - |
| hybrid | 87% | 0.84 | 1.00 | 0.76 | - | - |
| dense -> rerank[bge-reranker-v2-m3, 10] | 100% | 1.00 | 1.00 | 1.00 | 15129 | 16918 |
| dense -> rerank[bge-reranker-v2-m3, 20] | 100% | 1.00 | 1.00 | 1.00 | 25121 | 26917 |
| hybrid -> rerank[bge-reranker-v2-m3, 10] | 93% | 0.93 | 1.00 | 0.90 | 15259 | 16325 |
| hybrid -> rerank[bge-reranker-v2-m3, 20] | 100% | 1.00 | 1.00 | 1.00 | 26426 | 28062 |
| dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 10] | 100% | 0.89 | 1.00 | 0.83 | 706 | 875 |
| dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 20] | 100% | 0.88 | 1.00 | 0.82 | 1257 | 1329 |
| hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 10] | 93% | 0.87 | 1.00 | 0.80 | 726 | 795 |
| hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 20] | 100% | 0.89 | 1.00 | 0.83 | 1357 | 1454 |

## Models

| Model | Parameters | Load time (s) |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | 3.1 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | 1828.3 |

## Latency budget of the best pipeline (mean per question, ms)

| Stage | Measured (ms) | SLA (ms) | Status |
|---|---|---|---|
| 1. Query encoding (BGE-M3) | 328.3 | < 30 | over SLA |
| 2. Dense search Qdrant (HNSW) | 11.0 | < 40 | OK |
| 3. Sparse search BM25 (in memory) | 2.2 | < 15 | OK |
| 3b. Sparse search BGE-M3 (Qdrant) | 8.4 | < 15 | OK |
| 4. RRF fusion | 0.1 | < 5 | OK |
| 5. Cross-encoder reranking (bge-reranker-v2-m3, 10) | 15128.7 | < 150 | over SLA |

## Before / after reranking (evaluation sheet, section 2)

Question: "What is the withdrawal period after the delivery of equipment bought remotely with an Orange subscription?"

| Rank | Dense search only | Cosine | After reranking (best pipeline) | Rank before | Cross-encoder |
|:-:|---|:-:|---|:-:|:-:|
| 1 | Conditions générales de vente Mobicarte / Préambule (suite 3) | 0.7193 | (CGA) Services Internet et Mobiles / Article 5. Description du service | 7 | 0.740 |
| 2 | (CS) Changement de mobile / Article 4. Droit de rétractation en cas... | 0.7192 | (CS) Changement de mobile / Article 4. Droit de rétractation en cas... | 2 | 0.667 |
| 3 | (CGU) Option Illimité ligne Livebox vers mobiles / Article 9. Vente... | 0.7049 | Conditions générales de vente Mobicarte / Préambule (suite 3) | 1 | 0.585 |
| 4 | (CG) Prestations d’assistance à distance / Article 10. DROIT DE RÉT... | 0.6990 | (CGU) Option Sauvegarde en ligne / Article 12. Vente à distance | 6 | 0.480 |
| 5 | (CS) Prestations démarrage Smartphone pour les clients Origami Jet ... | 0.6983 | (CG) Prestations d’assistance à distance / Article 10. DROIT DE RÉT... | 4 | 0.441 |

## Rank of the first correct result per question (- = not in the top 10)

| # | Question | dense | hybrid | dense -> rerank[bge-reranker-v2-m3, 10] | dense -> rerank[bge-reranker-v2-m3, 20] | hybrid -> rerank[bge-reranker-v2-m3, 10] | hybrid -> rerank[bge-reranker-v2-m3, 20] | dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 10] | dense -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 20] | hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 10] | hybrid -> rerank[mmarco-mMiniLMv2-L12-H384-v1, 20] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Quel est le prix mensuel de l'offre 4G Home 200 Go ? | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 3 | 2 | 2 |
| 2 | Combien coûte le forfait 500 Go 5G+ pour un client q... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3 | Le prix de la location de la Livebox est-il inclus d... | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4 | Le forfait 500 Go 5G+ est-il disponible sans engagem... | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 5 | Y a-t-il des frais de résiliation pour la 4G Home ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | Quelle est la durée d'engagement de l'offre Série Sp... | 1 | 9 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
| 7 | Comment résilier l'option internet tranquillité 50 Mo ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | What is the withdrawal period after the delivery of ... | 7 | - | 1 | 1 | - | 1 | 3 | 3 | - | 3 |
| 9 | How many days do I have to withdraw after ordering a... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 10 | What is the maximum 4G+ download speed on the 500 GB... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 11 | Le prix de la location Livebox est-il inclus dans l'... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 12 | Quel est le montant des frais de construction pour u... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 13 | Quel est le prix de la Mobicarte Start 150Go ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 14 | Quel est le délai de rétractation de l'option Family... | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 15 | Quel est le prix mensuel de l'offre Mon Optimale Int... | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

## Reading

**Does reranking fix the order? Yes.** A cross-encoder reads the question and the chunk *together*, so it can tell apart RC summaries that share the same sentences. With bge-reranker-v2-m3 on the top 10 dense candidates, all 15 questions get the right chunk at rank 1: MRR@10 goes from 0.83 to 1.00 and Recall@5 from 93% to 100%. On the showcase question Q8 (English, withdrawal period), the CGA Article 5 chunk was #13 after fusion and comes first after reranking.

**Hybrid no longer helps once a reranker is added.** dense -> rerank scores as well as or better than hybrid -> rerank in every configuration (N=10: 1.00 vs 0.93 with bge, 0.89 vs 0.87 with mmarco). With 10 candidates the hybrid list loses one question completely (Recall@5 93%): fusion pushes some correct chunks below rank 10, where the reranker never sees them. With 20 candidates both first stages reach 1.00, but reranking costs twice as much.

**Quality vs latency (CPU laptop, dense first stage, 10 candidates):**

| Reranker | MRR@10 | Recall@5 | Rerank mean |
|---|---|---|---|
| bge-reranker-v2-m3 (568M) | 1.00 | 100% | 15.1 s |
| bge-reranker-v2-m3 int8 | 0.97 | 100% | 7.4 s |
| mmarco-mMiniLMv2 (118M) | 0.89 | 100% | 0.71 s |
| mmarco-mMiniLMv2 int8 | 0.90 | 93% | 0.37 s |

- `max_length` 256 / 384 / 512 changes nothing: the pairs are already shorter than 256 tokens. The cost comes from the size of the model.
- int8 quantization halves the time but loses one question (out of the top 5 for mmarco, off rank 1 for bge). Rejected (details in `benchmark_rerank_int8_s3.md`).

**Decision:**
- **Live pipeline:** dense top 10 -> mmarco-mMiniLMv2 (fp32). It keeps Recall@5 = 100% (the LLM receives the top 5: a chunk outside it cannot be cited) for ~0.7 s.
- **Quality mode (offline evaluation, Ragas):** dense top 10 -> bge-reranker-v2-m3 (fp32), MRR 1.00.

**Latency budget of the live pipeline (mean per question):**

| Stage | Measured | SLA | Status |
|---|---|---|---|
| Query encoding (BGE-M3) | ~360 ms | < 30 ms | over |
| Dense search (Qdrant HNSW) | 10-20 ms | < 40 ms | OK |
| BM25 / RRF (optional lanes) | 1-3 ms / 0.1 ms | < 15 / < 5 ms | OK |
| Reranking (mmarco, 10 pairs) | ~710 ms (p95 875) | < 150 ms | over |
| LLM time to first token | step 3.4 | < 600 ms | - |

**Limits and failure analysis:**
- 15 questions only: each difference between configurations is one question, and 1.00 is a ceiling, not a proof. The gold set should be extended, especially with exact-term questions, the case where hybrid helps.
- The SLAs assume a GPU. On this CPU-only laptop the two transformer stages are over budget (encoding ~12x, reranking ~5x with mmarco, ~100x with bge), while the search stages (Qdrant, BM25, RRF) are well within it. Remedies: a GPU (a cross-encoder scores 10 pairs in a few tens of ms there), a hosted reranking API, or caching frequent questions.
