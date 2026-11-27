# Hybrid retrieval benchmark (session 3)

Date: 2026-11-27 19:33 | gold set: `eval/gold/retrieval_s2.csv` (15 questions) | collection: telixia_docs (children 256 tokens) | RRF k tested: 60

Recall@20 = candidate recall: is the answer among the 20 chunks handed to the reranker?

| Mode | Recall@5 | MRR@10 | DocHit@5 | Recall@20 |
|---|---|---|---|---|
| dense | 93% | 0.83 | 93% | 100% |
| bm25 | 60% | 0.55 | 80% | 80% |
| sparse | 67% | 0.59 | 80% | 87% |
| rrf(dense+bm25) | 87% | 0.72 | 93% | 100% |
| rrf(dense+bm25_la) | 87% | 0.77 | 93% | 100% |
| rrf(dense+sparse) | 87% | 0.84 | 93% | 100% |
| rrf(dense+bm25_la+sparse) | 87% | 0.84 | 93% | 100% |

## By question category

| Mode | exact R@5 | exact MRR | paraphrase R@5 | paraphrase MRR |
|---|---|---|---|---|
| dense | 100% | 0.90 | 90% | 0.80 |
| bm25 | 100% | 1.00 | 40% | 0.33 |
| sparse | 100% | 0.87 | 50% | 0.45 |
| rrf(dense+bm25) | 100% | 1.00 | 80% | 0.59 |
| rrf(dense+bm25_la) | 100% | 1.00 | 80% | 0.65 |
| rrf(dense+sparse) | 100% | 1.00 | 80% | 0.77 |
| rrf(dense+bm25_la+sparse) | 100% | 1.00 | 80% | 0.76 |

## Latency per lane (mean over the questions, ms)

| Stage | Mean (ms) |
|---|---|
| encode | 439.5 |
| dense | 11.0 |
| sparse | 11.8 |
| bm25 | 2.3 |

## Rank of the first correct result (- = not in the top 20)

| # | Question | dense | bm25 | sparse | rrf(dense+bm25) | rrf(dense+bm25_la) | rrf(dense+sparse) | rrf(dense+bm25_la+sparse) |
|---|---|---|---|---|---|---|---|---|
| 1 | Quel est le prix mensuel de l'offre 4G Home 200 Go ? | 1 | 12 | 7 | 4 | 4 | 2 | 2 |
| 2 | Combien coûte le forfait 500 Go 5G+ pour un client qui a ... | 1 | 1 | 5 | 1 | 1 | 1 | 1 |
| 3 | Le prix de la location de la Livebox est-il inclus dans l... | 2 | 2 | 1 | 1 | 1 | 1 | 1 |
| 4 | Le forfait 500 Go 5G+ est-il disponible sans engagement ? | 3 | 2 | 1 | 2 | 2 | 1 | 1 |
| 5 | Y a-t-il des frais de résiliation pour la 4G Home ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | Quelle est la durée d'engagement de l'offre Série Spécial... | 1 | - | 17 | 8 | 8 | 6 | 9 |
| 7 | Comment résilier l'option internet tranquillité 50 Mo ? | 1 | 6 | 1 | 2 | 2 | 1 | 1 |
| 8 | What is the withdrawal period after the delivery of equip... | 7 | - | - | 13 | 7 | 13 | 13 |
| 9 | How many days do I have to withdraw after ordering a phon... | 1 | - | - | 1 | 1 | 1 | 1 |
| 10 | What is the maximum 4G+ download speed on the 500 GB 5G+ ... | 1 | 8 | 8 | 2 | 1 | 1 | 1 |
| 11 | Le prix de la location Livebox est-il inclus dans l'offre... | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 12 | Quel est le montant des frais de construction pour un rac... | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 13 | Quel est le prix de la Mobicarte Start 150Go ? | 1 | 1 | 3 | 1 | 1 | 1 | 1 |
| 14 | Quel est le délai de rétractation de l'option FamilyWall ... | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| 15 | Quel est le prix mensuel de l'offre Mon Optimale Internet ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

## Reading

_To complete: which lanes help, which RRF k, and why._
