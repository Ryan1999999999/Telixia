# Retrieval benchmark (session 2 deliverable)

Date: 2026-11-27 00:29 | gold set: `eval/gold/retrieval_s2.csv` (10 questions, 3 in English) | model: BGE-M3 | parents: 1024 tokens | overlap: 40 tokens

A hit = expected document **and** expected section. DocHit@5 = document only.

| Children | Mode | Recall@5 | MRR@10 | DocHit@5 | Chunks | Avg tokens | Search (ms) |
|---|---|---|---|---|---|---|---|
| 256 | dense | 90% | 0.80 | 90% | 3609 | 162 | 17 |
| 256 | hybrid | 90% | 0.85 | 90% | 3609 | 162 | 19 |
| 512 | dense | 90% | 0.78 | 90% | 2527 | 222 | 13 |
| 512 | hybrid | 90% | 0.78 | 90% | 2527 | 222 | 15 |

## Rank of the first correct result per question (- = not in the top 10)

| # | Question | 256 dense | 256 hybrid | 512 dense | 512 hybrid |
|---|---|---|---|---|---|
| 1 | Quel est le prix mensuel de l'offre 4G Home 200 Go ? | 1 | 1 | 1 | 1 |
| 2 | Combien coûte le forfait 500 Go 5G+ pour un client qui a une offre ... | 1 | 1 | 1 | 1 |
| 3 | Le prix de la location de la Livebox est-il inclus dans l'offre Liv... | 2 | 1 | 1 | 1 |
| 4 | Le forfait 500 Go 5G+ est-il disponible sans engagement ? | 3 | 1 | 3 | 1 |
| 5 | Y a-t-il des frais de résiliation pour la 4G Home ? | 1 | 1 | 1 | 1 |
| 6 | Quelle est la durée d'engagement de l'offre Série Spéciale Livebox ... | 1 | 2 | 1 | 2 |
| 7 | Comment résilier l'option internet tranquillité 50 Mo ? | 1 | 1 | 1 | 1 |
| 8 | What is the withdrawal period after the delivery of equipment bough... | 7 | - | - | - |
| 9 | How many days do I have to withdraw after ordering a phone change r... | 1 | 1 | 2 | 4 |
| 10 | What is the maximum 4G+ download speed on the 500 GB 5G+ plan? | 1 | 1 | 1 | 1 |

## Reading

**Decision: child chunks of 256 tokens (parents 1024) with hybrid search (dense + sparse, RRF fusion).**

1. **Recall@5 is the same (90%) for every variant.** The chunk size changes *where* the answer ranks, not *whether* it is found.
2. **256 ranks the answer higher:** MRR@10 0.80-0.85 vs 0.78 for 512. A 512-token chunk mixes several topics
   (about 225 vs 160 tokens on average), which dilutes its vector.
3. **Hybrid helps with 256 (+0.05 MRR).** It moves Q3 (Livebox Max vs Livebox Up: same sentence, only the offer
   name differs) and Q4 (exact terms "500 Go 5G+", "engagement") to rank 1, but costs one rank on Q6 and Q9, where
   the 64 contract summaries share the same vocabulary. Net effect positive; no gain with 512.
4. **Cost is negligible:** 3,609 vs 2,527 vectors (about 15 MB vs 10 MB of dense vectors); search stays at 10-20 ms.
5. **Remaining failure, Q8:** the CGA rule (14 days from the delivery of the equipment) is buried under about ten
   passages describing the 21-day rule for remote subscriptions, with almost identical wording. This is a typical
   case for a cross-encoder reranker (session 3).

### Lessons learned while benchmarking

- **The gold set was part of the problem.** Version 1 had two ambiguous questions: the Livebox rental price appears
  in 16 documents, and "withdrawal period" has two valid answers (14 vs 21 days). Rewriting them so that each has a
  single unambiguous source raised Recall@5 from 80% to 90%: a measurement error, not an engine error.
- **Data quality fixes found on the way:** 140 unreadable chunks dropped (a PDF with a broken font encoding,
  "(CGU) Sauvegarde mobile", plus empty fill-in lines and stray labels); language detected per chunk because the
  7 Orange Holiday contract summaries are bilingual (English + French in the same PDF).
- **Fair comparison:** same model (BGE-M3), same overlap (40), same parents (1024), same questions; only the child
  size changes. An embedding cache makes a full re-run take seconds instead of 38 minutes.

### Limits and next steps

- 10 questions is a small sample: one question = 10 points of Recall@5, and a 0.05 MRR gap is about one question
  moving by one rank. The results are indicative, not statistically significant.
- Next: extend the gold set to 30-50 questions; add a cross-encoder reranker (BAAI/bge-reranker-v2-m3) on the top-50
  hybrid candidates; evaluate generated answers with Ragas (faithfulness, answer relevance).
