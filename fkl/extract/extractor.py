"""Per-chunk fact extraction, evidence verification and storage.

The contract enforced here, and the reason the rest of the system can be trusted:

    A fact is stored only if its quote can be located, character-exact, in the
    text the model was actually shown.

Everything else - the schema, the vocabulary, the number of facts per chunk - is
allowed to vary freely with the documents. Grounding is the one thing that does
not bend, because it is what turns "the model said so" into "the document says
so, at this offset on this page".
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..config import Config
from ..db import transaction
from ..llm.base import LLMClient
from ..llm.deterministic import deterministic_facts
from ..llm.json_guard import GuardResult, guarded_json
from .grounding import locate_in_page
from .normalize import canonical_key, canonical_text, fact_type_name, normalize_value
from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt

# Keys the model is asked for; anything else it returns is kept verbatim in the
# payload. This list exists only to separate "core" from "emergent" for display.
CORE_KEYS = (
    "subject", "attribute", "value", "unit", "time_scope",
    "qualifier", "confidence", "source_quote",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ExtractionStats:
    chunks_seen: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    facts_proposed: int = 0
    facts_stored: int = 0
    rejected_ungrounded: int = 0
    rejected_invalid: int = 0
    rejected_duplicate: int = 0
    repair_outcomes: Counter = field(default_factory=Counter)
    match_modes: Counter = field(default_factory=Counter)
    elapsed_s: float = 0.0

    def merge(self, other: "ExtractionStats") -> None:
        self.chunks_seen += other.chunks_seen
        self.chunks_processed += other.chunks_processed
        self.chunks_failed += other.chunks_failed
        self.facts_proposed += other.facts_proposed
        self.facts_stored += other.facts_stored
        self.rejected_ungrounded += other.rejected_ungrounded
        self.rejected_invalid += other.rejected_invalid
        self.rejected_duplicate += other.rejected_duplicate
        self.repair_outcomes.update(other.repair_outcomes)
        self.match_modes.update(other.match_modes)
        self.elapsed_s += other.elapsed_s

    @property
    def grounding_rate(self) -> float:
        return self.facts_stored / self.facts_proposed if self.facts_proposed else 0.0

    @property
    def repair_rate(self) -> float:
        total = sum(self.repair_outcomes.values())
        if not total:
            return 0.0
        repaired = total - self.repair_outcomes.get("ok_first_try", 0)
        return repaired / total


def _fact_id(document_id: str, chunk_id: str, payload: dict, quote: str) -> str:
    """Deterministic id, so re-extracting the same chunk cannot duplicate facts."""
    basis = "|".join(
        [
            document_id,
            chunk_id,
            str(payload.get("subject")),
            str(payload.get("attribute")),
            str(payload.get("value")),
            str(payload.get("time_scope")),
            quote[:160],
        ]
    )
    return "fact_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _valid_shape(record: dict) -> bool:
    """Minimum viable fact: something it is about, and something it says."""
    if not isinstance(record, dict):
        return False
    has_identity = bool(str(record.get("subject") or "").strip()) or bool(
        str(record.get("attribute") or record.get("predicate") or "").strip()
    )
    has_quote = bool(str(record.get("source_quote") or "").strip())
    return has_identity and has_quote


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(conf, 0.0), 1.0)


def _log_repair(conn: sqlite3.Connection, chunk_id: str, result: GuardResult) -> None:
    conn.execute(
        """INSERT INTO repair_log (chunk_id, stage, attempts, outcome, error, raw_output, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            chunk_id,
            "extraction",
            result.attempts,
            result.outcome,
            (result.error or "")[:500],
            (result.raw or "")[:2000] if result.outcome != "ok_first_try" else None,
            _now(),
        ),
    )


def _upsert_fact_type(conn: sqlite3.Connection, name: str, document_id: str,
                      payload: dict, fact_id: str) -> None:
    """Grow the dynamic schema registry.

    A previously unseen `attribute` simply becomes a new row. A previously seen
    one accumulates any new payload keys into `observed_keys_json`, which is how
    shape drift within a fact type becomes visible without a migration.
    """
    row = conn.execute("SELECT observed_keys_json, fact_count FROM fact_types WHERE name = ?",
                       (name,)).fetchone()
    incoming = sorted(payload.keys())
    if row is None:
        conn.execute(
            """INSERT INTO fact_types
               (name, first_seen_document_id, first_seen_at, last_seen_at,
                fact_count, observed_keys_json, sample_fact_id)
               VALUES (?,?,?,?,?,?,?)""",
            (name, document_id, _now(), _now(), 1, json.dumps(incoming), fact_id),
        )
        return

    try:
        known = set(json.loads(row["observed_keys_json"]))
    except (ValueError, TypeError):
        known = set()
    merged = sorted(known.union(incoming))
    conn.execute(
        """UPDATE fact_types
              SET fact_count = fact_count + 1, last_seen_at = ?, observed_keys_json = ?
            WHERE name = ?""",
        (_now(), json.dumps(merged), name),
    )


def propose_records(
    chunk_text: str,
    client: LLMClient | None,
    cfg: Config,
    *,
    document_title: str | None = None,
) -> tuple[list[dict], GuardResult | None, str]:
    """Ask the backend for candidate facts. Pure inference - no database access.

    Split out from storage so the expensive half can run in a worker process
    while SQLite stays single-writer in the parent (see extract/runner.py).
    """
    if client is None:
        return deterministic_facts(chunk_text, document_title=document_title), None, "deterministic"

    result = guarded_json(
        client,
        EXTRACTION_SYSTEM,
        build_extraction_prompt(chunk_text, document_title=document_title),
        max_retries=cfg.json_max_retries,
        max_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
    )
    return (result.data or []), result, client.name


def store_records(
    conn: sqlite3.Connection,
    chunk: sqlite3.Row,
    page_text: str,
    records: list[dict],
    extractor_name: str,
) -> ExtractionStats:
    """Ground, normalise and persist proposed facts. Runs in the parent only."""
    stats = ExtractionStats(chunks_seen=1, chunks_processed=1)
    t0 = time.perf_counter()
    stats.facts_proposed = len(records)

    for record in records:
        if not _valid_shape(record):
            stats.rejected_invalid += 1
            continue

        quote = str(record.get("source_quote") or "").strip()
        match = locate_in_page(page_text, chunk["text"], chunk["char_start"], quote)
        if match is None:
            # The model produced text that is not in the document. Discard it.
            stats.rejected_ungrounded += 1
            continue

        payload = {k: v for k, v in record.items() if v is not None or k in CORE_KEYS}
        payload["source_quote"] = match.text  # store what the document really says

        fact_id = _fact_id(chunk["document_id"], chunk["id"], payload, match.text)
        if conn.execute("SELECT 1 FROM facts WHERE id = ?", (fact_id,)).fetchone():
            stats.rejected_duplicate += 1
            continue

        value_num, unit = normalize_value(record.get("value"), record.get("unit"))
        type_name = fact_type_name(payload)

        conn.execute(
            """INSERT INTO facts
               (id, document_id, chunk_id, payload_json, fact_type, canonical_key,
                canonical_text, value_num, unit, time_scope, confidence, extractor,
                embedded, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (
                fact_id,
                chunk["document_id"],
                chunk["id"],
                json.dumps(payload, ensure_ascii=False),
                type_name,
                canonical_key(record.get("subject"), record.get("attribute") or record.get("predicate")),
                canonical_text(payload),
                value_num,
                unit,
                str(record.get("time_scope")) if record.get("time_scope") else None,
                _coerce_confidence(record.get("confidence")),
                extractor_name,
                _now(),
            ),
        )
        conn.execute(
            """INSERT INTO evidence
               (id, fact_id, document_id, chunk_id, pdf_page_index, printed_page_label,
                quote, quote_char_start, quote_char_end, match_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                f"ev_{fact_id[5:]}",
                fact_id,
                chunk["document_id"],
                chunk["id"],
                chunk["pdf_page_index"],
                chunk["printed_page_label"],
                match.text,
                match.start,
                match.end,
                match.match_mode,
            ),
        )
        _upsert_fact_type(conn, type_name, chunk["document_id"], payload, fact_id)

        stats.facts_stored += 1
        stats.match_modes[match.match_mode] += 1

    stats.elapsed_s = time.perf_counter() - t0
    return stats


def extract_from_chunk(
    conn: sqlite3.Connection,
    chunk: sqlite3.Row,
    page_text: str,
    client: LLMClient | None,
    cfg: Config,
    *,
    document_title: str | None = None,
) -> ExtractionStats:
    """Single-process path: infer, log the JSON outcome, then store."""
    records, guard, extractor_name = propose_records(
        chunk["text"], client, cfg, document_title=document_title
    )

    if guard is not None:
        _log_repair(conn, chunk["id"], guard)
        if not guard.ok:
            stats = ExtractionStats(chunks_seen=1, chunks_failed=1)
            stats.repair_outcomes[guard.outcome] += 1
            mark_chunk_done(conn, chunk["id"], "failed", 0)
            return stats

    stats = store_records(conn, chunk, page_text, records, extractor_name)
    stats.repair_outcomes[guard.outcome if guard else "ok_first_try"] += 1
    mark_chunk_done(conn, chunk["id"], "done", stats.facts_stored)
    return stats


def mark_chunk_done(
    conn: sqlite3.Connection, chunk_id: str, status: str, facts: int
) -> None:
    """Record that a chunk has been attempted, so a re-run can resume.

    Resumability matters here because a full-corpus extraction takes ~an hour on
    CPU; losing all of it to one interruption would be unacceptable.
    """
    conn.execute(
        """INSERT INTO extraction_progress (chunk_id, status, facts, processed_at)
           VALUES (?,?,?,?)
           ON CONFLICT(chunk_id) DO UPDATE SET
             status=excluded.status, facts=excluded.facts, processed_at=excluded.processed_at""",
        (chunk_id, status, facts, _now()),
    )


def pending_chunks(
    conn: sqlite3.Connection, cfg: Config, *, document_id: str | None = None
) -> list[sqlite3.Row]:
    """Extraction chunks not yet attempted, interleaved across documents.

    Ordering matters more than it looks. Processing document by document means a
    partial run - and on CPU a full run takes about an hour, so partial runs are
    the common case - produces facts from only the first document or two, and
    therefore *zero* cross-document relationships. The whole point of the system
    is the links between documents, so a run that is interrupted halfway should
    still be able to demonstrate them.

    Round-robin fixes that: take the first chunk of every document, then the
    second of every document, and so on. Any prefix of the resulting order covers
    all documents roughly evenly.
    """
    sql = """SELECT c.*, p.text AS page_text, d.title AS document_title,
                    ROW_NUMBER() OVER (PARTITION BY c.document_id ORDER BY c.ordinal) AS doc_rank
               FROM chunks c
               JOIN pages p ON p.id = c.page_id
               JOIN documents d ON d.id = c.document_id
              WHERE c.role = 'extraction'
                AND c.numeric_density >= ?
                AND NOT EXISTS (SELECT 1 FROM extraction_progress ep
                                 WHERE ep.chunk_id = c.id AND ep.status = 'done')"""
    params: list = [cfg.min_numeric_density]
    if document_id:
        sql += " AND c.document_id = ?"
        params.append(document_id)
    sql += " ORDER BY doc_rank, c.document_id"
    return conn.execute(sql, params).fetchall()


def extract_document(
    conn: sqlite3.Connection,
    document_id: str,
    client: LLMClient | None,
    cfg: Config,
    *,
    on_progress=None,
    commit_every: int = 1,
) -> ExtractionStats:
    """Extract facts for every eligible extraction chunk of one document."""
    doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if doc is None:
        raise ValueError(f"unknown document {document_id}")

    chunks = conn.execute(
        """SELECT c.*, p.text AS page_text
             FROM chunks c JOIN pages p ON p.id = c.page_id
            WHERE c.document_id = ? AND c.role = 'extraction'
              AND c.numeric_density >= ?
            ORDER BY c.ordinal""",
        (document_id, cfg.min_numeric_density),
    ).fetchall()

    stats = ExtractionStats()
    for index, chunk in enumerate(chunks, 1):
        stats.merge(
            extract_from_chunk(
                conn, chunk, chunk["page_text"], client, cfg, document_title=doc["title"]
            )
        )
        if index % commit_every == 0:
            conn.commit()
        if on_progress:
            on_progress(index, len(chunks), stats)

    with transaction(conn):
        conn.execute("UPDATE documents SET status = 'extracted' WHERE id = ?", (document_id,))
    return stats
