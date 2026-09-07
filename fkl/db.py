"""SQLite storage layer.

Design notes
------------
* `facts.payload_json` is deliberately schema-free. The LLM decides what keys a
  fact object carries; we never migrate a column to accommodate a new fact
  shape. The `fact_types` table is a *registry* that observes shapes as they
  appear, which is what makes the schema "emerge from the documents".
* Character offsets are stored against `pages.text` (the exact string PyMuPDF
  returned) so that any quoted span can be re-verified byte-for-byte later.
* SQLite is used with WAL and enforced foreign keys so that deleting a document
  cleanly removes its pages, chunks, facts, evidence and relationships.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,        -- doc_<sha256(file bytes)[:16]>
    content_sha256 TEXT NOT NULL UNIQUE,    -- full hash: the incremental-ingest key
    filename       TEXT NOT NULL,
    source_path    TEXT,
    title          TEXT,
    page_count     INTEGER,
    byte_size      INTEGER,
    parser_version TEXT,
    status         TEXT NOT NULL,           -- parsed | extracted | linked | error
    ingested_at    TEXT NOT NULL,
    meta_json      TEXT NOT NULL DEFAULT '{}'
);

-- -------------------------------------------------------------------- pages
-- `text` is the verbatim PyMuPDF output. All offsets elsewhere index into it.
CREATE TABLE IF NOT EXISTS pages (
    id                 TEXT PRIMARY KEY,    -- <doc_id>:p<index>
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    pdf_page_index     INTEGER NOT NULL,    -- 0-based physical page
    printed_page_label TEXT,                -- page number printed on the page, if detected
    text               TEXT NOT NULL,
    char_count         INTEGER NOT NULL,
    table_count        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (document_id, pdf_page_index)
);

-- ------------------------------------------------------------------- chunks
-- Two coexisting views of the same text, distinguished by `role`:
--   extraction -> non-overlapping units fed to the LLM (usually a whole page)
--   retrieval  -> overlapping windows used only for embedding/similarity
CREATE TABLE IF NOT EXISTS chunks (
    id                 TEXT PRIMARY KEY,
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id            TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    pdf_page_index     INTEGER NOT NULL,
    printed_page_label TEXT,
    role               TEXT NOT NULL,       -- extraction | retrieval
    kind               TEXT NOT NULL,       -- prose | table
    ordinal            INTEGER NOT NULL,    -- order within (document, role)
    char_start         INTEGER,             -- offset into pages.text
    char_end           INTEGER,
    span_verified      INTEGER NOT NULL DEFAULT 1,  -- 0 for synthesised table markdown
    text               TEXT NOT NULL,
    token_estimate     INTEGER NOT NULL,
    numeric_density    REAL NOT NULL DEFAULT 0.0,
    UNIQUE (document_id, role, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_role ON chunks(document_id, role);
CREATE INDEX IF NOT EXISTS idx_chunks_page     ON chunks(page_id);

-- -------------------------------------------------------------------- facts
-- No fixed columns for the fact body: `payload_json` holds whatever shape the
-- extractor produced. The promoted columns below exist only for indexing.
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    payload_json  TEXT NOT NULL,
    fact_type     TEXT,        -- canonical attribute name; keys the fact_types registry
    canonical_key TEXT,        -- normalised subject + attribute, for grouping
    canonical_text TEXT,       -- the string that gets embedded
    value_num     REAL,        -- normalised numeric value, when the fact has one
    unit          TEXT,        -- normalised unit
    time_scope    TEXT,
    confidence    REAL,
    extractor     TEXT NOT NULL,  -- local-llm | groq | deterministic
    embedded      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_doc       ON facts(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_type      ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_canonical ON facts(canonical_key);

-- ----------------------------------------------------------------- evidence
-- Mandatory. A fact with no verifiable evidence row is discarded at extraction
-- time, which is our main defence against hallucinated numbers.
CREATE TABLE IF NOT EXISTS evidence (
    id                 TEXT PRIMARY KEY,
    fact_id            TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id           TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    pdf_page_index     INTEGER NOT NULL,
    printed_page_label TEXT,
    quote              TEXT NOT NULL,   -- exact substring of the source text
    quote_char_start   INTEGER,         -- offset into pages.text
    quote_char_end     INTEGER,
    match_mode         TEXT NOT NULL    -- exact | normalised_whitespace
);
CREATE INDEX IF NOT EXISTS idx_evidence_fact ON evidence(fact_id);

-- --------------------------------------------------------------- fact_types
-- The dynamic schema registry: one row per distinct fact "shape" observed.
CREATE TABLE IF NOT EXISTS fact_types (
    name                  TEXT PRIMARY KEY,
    first_seen_document_id TEXT,
    first_seen_at         TEXT NOT NULL,
    last_seen_at          TEXT NOT NULL,
    fact_count            INTEGER NOT NULL DEFAULT 0,
    observed_keys_json    TEXT NOT NULL DEFAULT '[]',  -- union of payload keys seen
    sample_fact_id        TEXT
);

-- ------------------------------------------------------------ relationships
-- First-class, queryable records - not merely a graph rendering.
CREATE TABLE IF NOT EXISTS relationships (
    id               TEXT PRIMARY KEY,
    fact_a_id        TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    fact_b_id        TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    relation_type    TEXT NOT NULL,  -- CORROBORATES|CONTRADICTS|CONTEXT_RECONCILED|UNRELATED
    reason_tag       TEXT,           -- units_differ | different_period | different_scope | ...
    explanation      TEXT,
    confidence       REAL,
    similarity_score REAL,
    cross_document   INTEGER NOT NULL DEFAULT 1,
    model            TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE (fact_a_id, fact_b_id)
);
CREATE INDEX IF NOT EXISTS idx_rel_a    ON relationships(fact_a_id);
CREATE INDEX IF NOT EXISTS idx_rel_b    ON relationships(fact_b_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);

-- --------------------------------------------------------------------- jobs
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    document_id TEXT,
    kind        TEXT NOT NULL,   -- ingest | extract | link
    status      TEXT NOT NULL,   -- queued | running | done | error
    progress    REAL NOT NULL DEFAULT 0.0,
    message     TEXT,
    started_at  TEXT,
    finished_at TEXT
);

-- --------------------------------------------------------------- repair_log
-- Powers the "how often did the JSON repair path fire?" metric in DECISIONS.md.
CREATE TABLE IF NOT EXISTS repair_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id   TEXT,
    stage      TEXT NOT NULL,   -- extraction | relation
    attempts   INTEGER NOT NULL DEFAULT 1,
    outcome    TEXT NOT NULL,   -- ok_first_try|ok_after_reprompt|ok_after_repair|failed
    error      TEXT,
    raw_output TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repair_outcome ON repair_log(outcome);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (and if necessary initialise) the database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block atomically; roll everything back on any exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def json_load(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    """Remove a document and everything derived from it (FK cascade does the work)."""
    with transaction(conn):
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in (
        "documents",
        "pages",
        "chunks",
        "facts",
        "evidence",
        "fact_types",
        "relationships",
    ):
        out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return out
