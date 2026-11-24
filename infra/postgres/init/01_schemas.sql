CREATE SCHEMA IF NOT EXISTS analytics;  -- customers (IBM Telco)
CREATE SCHEMA IF NOT EXISTS catalog;    -- Orange offers extracted from PDFs
CREATE SCHEMA IF NOT EXISTS market;     -- Orange KPIs, ARCEP / INT market data
CREATE SCHEMA IF NOT EXISTS docs;       -- document registry (traceability)
CREATE SCHEMA IF NOT EXISTS audit;      -- log of SQL queries run by the copilot

CREATE TABLE IF NOT EXISTS docs.documents (
    doc_id           TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    source_url       TEXT,
    file_path        TEXT NOT NULL,
    category         TEXT NOT NULL,
    doc_type         TEXT,
    country          TEXT DEFAULT 'FR',
    publication_date DATE,
    language         TEXT DEFAULT 'fr',
    is_synthetic     BOOLEAN DEFAULT FALSE,
    sha256           TEXT UNIQUE,
    n_pages          INT,
    n_chunks         INT,
    ingested_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS docs.parent_chunks (
    parent_id   TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES docs.documents(doc_id) ON DELETE CASCADE,
    section     TEXT,
    page_start  INT,
    page_end    INT,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc ON docs.parent_chunks(doc_id);

CREATE TABLE IF NOT EXISTS audit.sql_log (
    id            BIGSERIAL PRIMARY KEY,
    asked_at      TIMESTAMPTZ DEFAULT now(),
    question      TEXT NOT NULL,
    generated_sql TEXT,
    validated     BOOLEAN,
    rejection     TEXT,
    row_count     INT,
    duration_ms   INT
);