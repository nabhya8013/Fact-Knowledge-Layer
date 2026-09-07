"""The four required demonstration cases, selected by query rather than by hand.

The brief asks for one clear example each of a corroborated fact, a genuine
contradiction, an apparent contradiction explained by context, and a documented
extraction/reasoning failure. Nothing here is hardcoded to a document, an entity
or a figure: each case is a ranked query over whatever happens to be in the
store, so the same code produces a showcase for an entirely new corpus.

The failure case is deliberately *computed* too. Rather than narrating a story,
it surfaces the real rejection and repair counters the pipeline recorded, plus
concrete examples of quotes that failed to ground - so the reviewer sees the
system's actual failure modes rather than a curated anecdote. The written
analysis of those failures lives in DECISIONS.md.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import json_load

# A showcase example should be legible on its own, so prefer relationships whose
# two facts carry evidence and an explanation.
_BASE_SQL = """
SELECT r.*,
       fa.document_id AS a_doc, fb.document_id AS b_doc
  FROM relationships r
  JOIN facts fa ON fa.id = r.fact_a_id
  JOIN facts fb ON fb.id = r.fact_b_id
 WHERE r.relation_type = ?
   AND r.cross_document = 1
   AND fa.document_id != fb.document_id
   AND COALESCE(TRIM(r.explanation), '') != ''
"""


def _fact_detail(conn: sqlite3.Connection, fact_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT f.*, d.filename, d.title AS document_title,
                  e.quote, e.pdf_page_index, e.printed_page_label, e.match_mode,
                  e.quote_char_start, e.quote_char_end
             FROM facts f
             JOIN documents d ON d.id = f.document_id
             LEFT JOIN evidence e ON e.fact_id = f.id
            WHERE f.id = ? LIMIT 1""",
        (fact_id,),
    ).fetchone()
    if row is None:
        return None

    payload = json_load(row["payload_json"], {}) or {}
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "document": row["filename"],
        "document_title": row["document_title"],
        "subject": payload.get("subject"),
        "attribute": payload.get("attribute") or payload.get("predicate"),
        "value": payload.get("value"),
        "unit": payload.get("unit"),
        "time_scope": payload.get("time_scope"),
        "qualifier": payload.get("qualifier"),
        "fact_type": row["fact_type"],
        "value_num": row["value_num"],
        "normalised_unit": row["unit"],
        "confidence": row["confidence"],
        "extractor": row["extractor"],
        "payload": payload,
        "evidence": {
            "quote": row["quote"],
            "pdf_page_index": row["pdf_page_index"],
            "printed_page_label": row["printed_page_label"],
            "page_label": row["printed_page_label"] or f"pdf#{row['pdf_page_index']}",
            "match_mode": row["match_mode"],
            "char_start": row["quote_char_start"],
            "char_end": row["quote_char_end"],
        },
    }


def _relationship_example(
    conn: sqlite3.Connection, relation_type: str, *, limit: int = 3
) -> list[dict[str, Any]]:
    """Best examples of one relation type, strongest evidence first."""
    rows = conn.execute(
        _BASE_SQL + " ORDER BY r.confidence DESC, r.similarity_score DESC LIMIT ?",
        (relation_type, limit),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        a = _fact_detail(conn, row["fact_a_id"])
        b = _fact_detail(conn, row["fact_b_id"])
        if not a or not b or not a["evidence"]["quote"] or not b["evidence"]["quote"]:
            continue
        out.append(
            {
                "relation_type": row["relation_type"],
                "reason_tag": row["reason_tag"],
                "explanation": row["explanation"],
                "confidence": row["confidence"],
                "similarity_score": row["similarity_score"],
                "model": row["model"],
                "fact_a": a,
                "fact_b": b,
            }
        )
    return out


def failure_report(conn: sqlite3.Connection, *, sample_size: int = 5) -> dict[str, Any]:
    """The measured failure surface, from counters the pipeline actually recorded."""
    # `repair_log` holds one row per chunk whose JSON needed a re-prompt, a
    # mechanical repair, or failed outright - a clean first-try parse is *not*
    # logged. So the denominator for a repair rate is the number of chunks
    # attempted (each made at least one JSON call), not the size of repair_log.
    repair = dict(
        conn.execute("SELECT outcome, COUNT(*) FROM repair_log GROUP BY outcome").fetchall()
    )
    repaired = repair.get("ok_after_reprompt", 0) + repair.get("ok_after_repair", 0)
    json_failed = repair.get("failed", 0)

    chunks_attempted = conn.execute("SELECT COUNT(*) FROM extraction_progress").fetchone()[0]
    chunks_failed = conn.execute(
        "SELECT COUNT(*) FROM extraction_progress WHERE status = 'failed'"
    ).fetchone()[0]
    denom = chunks_attempted or 1

    match_modes = dict(
        conn.execute("SELECT match_mode, COUNT(*) FROM evidence GROUP BY match_mode").fetchall()
    )

    # Chunks that ran cleanly but yielded nothing: usually every proposed fact
    # failed to ground, which is the interesting case.
    barren = conn.execute(
        "SELECT COUNT(*) FROM extraction_progress WHERE status='done' AND facts = 0"
    ).fetchone()[0]

    samples = [
        dict(r)
        for r in conn.execute(
            """SELECT chunk_id, outcome, attempts, error, substr(raw_output, 1, 400) AS raw_output
                 FROM repair_log
                WHERE outcome IN ('failed', 'ok_after_repair')
                ORDER BY id DESC LIMIT ?""",
            (sample_size,),
        ).fetchall()
    ]

    # A concrete look at the two real failure modes, not just their counts.
    grounding_failures = [
        {"chunk_id": r["chunk_id"], "page_label": r["printed_page_label"],
         "text": (r["text"] or "")[:500]}
        for r in conn.execute(
            """SELECT ep.chunk_id, c.printed_page_label, c.text
                 FROM extraction_progress ep JOIN chunks c ON c.id = ep.chunk_id
                WHERE ep.status = 'done' AND ep.facts = 0
                ORDER BY ep.processed_at DESC LIMIT ?""",
            (sample_size,),
        ).fetchall()
    ]
    reconstructed_span_examples = [
        {"fact_id": r["fact_id"], "canonical_text": r["canonical_text"],
         "document": r["filename"], "page_label": r["printed_page_label"],
         "stored_span": (r["quote"] or "")[:400]}
        for r in conn.execute(
            """SELECT e.fact_id, e.quote, e.printed_page_label, f.canonical_text, d.filename
                 FROM evidence e
                 JOIN facts f ON f.id = e.fact_id
                 JOIN documents d ON d.id = e.document_id
                WHERE e.match_mode = 'reconstructed_span'
                ORDER BY length(e.quote) DESC LIMIT ?""",
            (sample_size,),
        ).fetchall()
    ]

    return {
        "json_calls_needing_repair": repaired + json_failed,
        "json_repair_rate": repaired / denom,
        "json_failure_rate": json_failed / denom,
        "json_outcomes": repair,
        "evidence_match_modes": match_modes,
        "chunks_attempted": chunks_attempted,
        "chunks_failed": chunks_failed,
        "chunks_yielding_no_grounded_fact": barren,
        "recent_repair_samples": samples,
        "grounding_failures": grounding_failures,
        "reconstructed_span_examples": reconstructed_span_examples,
        "note": (
            "Ungrounded facts are discarded rather than stored, so these counters "
            "are the visible surface of extraction failure. See DECISIONS.md for "
            "the narrative analysis of specific failures."
        ),
    }


def build_showcase(conn: sqlite3.Connection, *, per_case: int = 3) -> dict[str, Any]:
    """Assemble all four required demonstration cases."""
    cases = {
        "corroboration": {
            "requirement": "(a) a corroborated fact across documents",
            "relation_type": "CORROBORATES",
            "examples": _relationship_example(conn, "CORROBORATES", limit=per_case),
        },
        "contradiction": {
            "requirement": "(b) a genuine or likely contradiction",
            "relation_type": "CONTRADICTS",
            "examples": _relationship_example(conn, "CONTRADICTS", limit=per_case),
        },
        "context_reconciled": {
            "requirement": "(c) an apparent contradiction explained by context",
            "relation_type": "CONTEXT_RECONCILED",
            "examples": _relationship_example(conn, "CONTEXT_RECONCILED", limit=per_case),
        },
        "failure": {
            "requirement": "(d) a documented extraction or reasoning failure",
            "relation_type": None,
            "report": failure_report(conn),
        },
    }

    missing = [
        name
        for name, case in cases.items()
        if case.get("relation_type") and not case["examples"]
    ]
    return {
        "cases": cases,
        "complete": not missing,
        "missing": missing,
        "totals": {
            "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "facts": conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
            "relationships": conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
        },
    }
