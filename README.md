
## Backlog: planned extensions (after the core)
- **Arabizi / Tunisian Derja mode**: LLM query rewriting (Arabizi -> French) before retrieval,
  evaluated on 10 Arabizi questions (Recall@5 with vs. without rewriting).

## Team & responsibilities

Solo project: **Rayen Sghairi** (M2 AI & Big Data / BI).

| Area | Responsible | Status |
|---|---|---|
| Framing, data sourcing, documentation | Rayen Sghairi | done (S1) |
| Infrastructure (Docker Compose, PostgreSQL, Qdrant) | Rayen Sghairi | done (S2) |
| Ingestion, chunking, embeddings | Rayen Sghairi | done (S2) |
| Retrieval (hybrid, reranker, evaluation) | Rayen Sghairi | S3 |
| Text-to-SQL, KPI layer | Rayen Sghairi | S4 |
| API, router, UI, CI/CD | Rayen Sghairi | S5-S7 |

## Ingestion (standalone)

    python scripts/ingest.py            # downloads the corpus if missing, builds chunks if stale, indexes
    python scripts/search_test.py --demo

Result: 147 documents -> 1,592 parents -> 3,749 chunks (612 tables), 936 boilerplate blocks
and 1,071 boilerplate lines removed. Deterministic: same corpus, same chunk ids.
