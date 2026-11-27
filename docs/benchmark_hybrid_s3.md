# Hybrid retrieval benchmark (session 3)

Date: 2026-11-27 01:15 | gold set: `eval/gold/retrieval_s2.csv` (10 questions) | collection: telixia_docs (children 256 tokens) | RRF k tested: 10, 60

Recall@20 = candidate recall: is the answer among the 20 chunks handed to the reranker?

| Mode | Recall@5 | MRR@10 | DocHit@5 | Recall@20 |
|---|---|---|---|---|
| dense | 90% | 0.80 | 90% | 100% |
| bm25 | 40% | 0.33 | 70% | 70% |
| sparse | 50% | 0.45 | 70% | 80% |
| rrf(dense+bm25) k=10 | 90% | 0.59 | 90% | 100% |
| rrf(dense+bm25) k=60 | 80% | 0.59 | 90% | 100% |
| rrf(dense+bm25_la) k=10 | 90% | 0.66 | 90% | 100% |
| rrf(dense+bm25_la) k=60 | 80% | 0.65 | 90% | 100% |
| rrf(dense+sparse) k=10 | 90% | 0.78 | 90% | 100% |
| rrf(dense+sparse) k=60 | 80% | 0.77 | 90% | 100% |
| rrf(dense+bm25_la+sparse) k=10 | 80% | 0.77 | 90% | 100% |
| rrf(dense+bm25_la+sparse) k=60 | 80% | 0.76 | 90% | 100% |

## Latency per lane (mean over the questions, ms)

| Stage | Mean (ms) |
|---|---|
| encode | 401.0 |
| dense | 9.8 |
| sparse | 9.8 |
| bm25 | 2.5 |

## Rank of the first correct result (- = not in the top 20)

| # | Question | dense | bm25 | sparse | rrf(dense+bm25) k=10 | rrf(dense+bm25) k=60 | rrf(dense+bm25_la) k=10 | rrf(dense+bm25_la) k=60 | rrf(dense+sparse) k=10 | rrf(dense+sparse) k=60 | rrf(dense+bm25_la+sparse) k=10 | rrf(dense+bm25_la+sparse) k=60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Quel est le prix mensuel de l'offre 4G Home 200 Go ? | 1 | 12 | 7 | 4 | 4 | 4 | 4 | 2 | 2 | 2 | 2 |
| 2 | Combien coûte le forfait 500 Go 5G+ pour un client qui a ... | 1 | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3 | Le prix de la location de la Livebox est-il inclus dans l... | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4 | Le forfait 500 Go 5G+ est-il disponible sans engagement ? | 3 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| 5 | Y a-t-il des frais de résiliation pour la 4G Home ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | Quelle est la durée d'engagement de l'offre Série Spécial... | 1 | - | 17 | 5 | 8 | 5 | 8 | 3 | 6 | 6 | 9 |
| 7 | Comment résilier l'option internet tranquillité 50 Mo ? | 1 | 6 | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| 8 | What is the withdrawal period after the delivery of equip... | 7 | - | - | 13 | 13 | 7 | 7 | 13 | 13 | 13 | 13 |
| 9 | How many days do I have to withdraw after ordering a phon... | 1 | - | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 10 | What is the maximum 4G+ download speed on the 500 GB 5G+ ... | 1 | 8 | 8 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |

## Reading

_To complete: which lanes help, which RRF k, and why._
