"""Classify candidate fact pairs into first-class relationship records.

The brief is explicit that a graph picture is not sufficient: relationships must
be structured, queryable data. So every classified pair becomes a row carrying
both fact ids, the relation type, a machine-readable reason tag, a
natural-language explanation and a confidence - queryable by type, by reason, by
either fact, or by document.

Two-step classification, deliberately
-------------------------------------
A deterministic pass runs first and computes what can be known without
judgement: do the canonical keys match, are the units comparable, do the
magnitudes agree, do the periods or scopes differ. That verdict is passed to the
LLM as a *hint* rather than used directly, because arithmetic cannot tell that
"revenue from contract with customers" and "revenue from operations" are the
same property, while the LLM cannot reliably tell that 8,142 crore equals 81.42
billion. Each covers the other's blind spot.

The deterministic verdict is also the fallback when no model is available, which
keeps linking working in the zero-model configuration.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..config import Config
from ..db import json_load, transaction
from ..extract.normalize import normalise_period, normalize_value, values_agree
from ..extract.prompts import RELATION_SYSTEM, build_relation_prompt
from ..llm.base import LLMClient
from ..llm.deterministic import deterministic_relation
from ..llm.json_guard import guarded_json
from .vector_index import Candidate, VectorIndex

VALID_RELATIONS = {"CORROBORATES", "CONTRADICTS", "CONTEXT_RECONCILED", "UNRELATED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LinkStats:
    candidates: int = 0
    classified: int = 0
    stored: int = 0
    skipped_existing: int = 0
    failed: int = 0
    by_type: Counter = field(default_factory=Counter)
    by_reason: Counter = field(default_factory=Counter)
    repair_outcomes: Counter = field(default_factory=Counter)
    elapsed_s: float = 0.0


def load_fact_view(conn: sqlite3.Connection, fact_id: str) -> dict | None:
    """A fact plus everything needed to judge and to cite it."""
    row = conn.execute(
        """SELECT f.*, d.title AS document_title, d.filename,
                  e.quote, e.pdf_page_index, e.printed_page_label
             FROM facts f
             JOIN documents d ON d.id = f.document_id
             LEFT JOIN evidence e ON e.fact_id = f.id
            WHERE f.id = ?
            LIMIT 1""",
        (fact_id,),
    ).fetchone()
    if row is None:
        return None

    payload = json_load(row["payload_json"], {}) or {}
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "document_title": row["document_title"] or row["filename"],
        "filename": row["filename"],
        "subject": payload.get("subject"),
        "attribute": payload.get("attribute") or payload.get("predicate"),
        "value": payload.get("value"),
        "unit": payload.get("unit"),
        "time_scope": payload.get("time_scope"),
        "qualifier": payload.get("qualifier"),
        "source_quote": row["quote"] or payload.get("source_quote"),
        "page": row["printed_page_label"] or f"pdf#{row['pdf_page_index']}",
        "value_num": row["value_num"],
        "canonical_key": row["canonical_key"],
    }


def _numeric_hint(a: dict, b: dict) -> str:
    """Arithmetic the model should not have to do, stated plainly."""
    va, ua = normalize_value(a.get("value"), a.get("unit"))
    vb, ub = normalize_value(b.get("value"), b.get("unit"))
    if va is None or vb is None:
        return "One or both values are not numeric, so no magnitude comparison is possible."
    if ua and ub and ua != ub:
        return (
            f"Normalised magnitudes are {va:g} {ua} and {vb:g} {ub}. The units differ, "
            "so these are not directly comparable."
        )
    agree = values_agree(va, vb)
    verdict = "the same magnitude" if agree else "different magnitudes"
    return f"Normalised to a common scale these are {va:g} and {vb:g} - {verdict}."


def build_pair_prompt(a: dict, b: dict) -> str:
    hint = _numeric_hint(a, b)
    rule = deterministic_relation(a, b)
    return (
        build_relation_prompt(a, b)
        + f"\n\nNumeric check: {hint}\n"
        + f"Rule-based first guess (may be wrong about wording): {rule['relation_type']} "
        + f"({rule['reason_tag']})\n"
    )


def _contradiction_blocker(a: dict, b: dict) -> tuple[str, str] | None:
    """Arithmetic the model must not be allowed to override.

    A genuine contradiction requires that everything except the value match. When
    the two facts are stated for different periods, or in units that are not
    directly comparable, the difference is explained by context - it cannot be a
    contradiction however the model phrases it. Returns (reason_tag, explanation)
    when a CONTRADICTS verdict should be downgraded, else None.
    """
    scope_a = normalise_period(a.get("time_scope"))
    scope_b = normalise_period(b.get("time_scope"))
    if scope_a and scope_b and scope_a != scope_b:
        return (
            "different_period",
            f"The two facts cover different periods "
            f"('{a.get('time_scope')}' versus '{b.get('time_scope')}'), so the "
            "difference in value is explained by context rather than a conflict.",
        )

    _, unit_a = normalize_value(a.get("value"), a.get("unit"))
    _, unit_b = normalize_value(b.get("value"), b.get("unit"))
    if unit_a and unit_b and unit_a != unit_b:
        return (
            "units_differ",
            f"The two facts are stated in units that are not directly comparable "
            f"({unit_a} versus {unit_b}); no conversion is applied, so this is not "
            "a contradiction.",
        )
    return None


def _corroboration_signal(a: dict, b: dict) -> str | None:
    """Arithmetic that positively confirms a corroboration.

    Same period, comparable units, and magnitudes that agree within tolerance:
    the two facts state the same quantity, no matter how differently they are
    worded. Returns an explanation when the model's weaker verdict should be
    lifted to CORROBORATES, else None.
    """
    scope_a = normalise_period(a.get("time_scope"))
    scope_b = normalise_period(b.get("time_scope"))
    if not scope_a or not scope_b or scope_a != scope_b:
        return None

    va, unit_a = normalize_value(a.get("value"), a.get("unit"))
    vb, unit_b = normalize_value(b.get("value"), b.get("unit"))
    if va is None or vb is None or (unit_a and unit_b and unit_a != unit_b):
        return None
    if values_agree(va, vb) is not True:
        return None
    return (
        f"Both facts state the same value ({va:g}{(' ' + unit_a) if unit_a else ''}) "
        f"for the same period ('{a.get('time_scope')}' / '{b.get('time_scope')}'), "
        "so they corroborate each other despite the different wording."
    )


def classify_pair(
    a: dict, b: dict, client: LLMClient | None, cfg: Config
) -> tuple[dict, str]:
    """Return (verdict, json_outcome). Falls back to rules with no model."""
    if client is None:
        return deterministic_relation(a, b), "ok_first_try"

    result = guarded_json(
        client,
        RELATION_SYSTEM,
        build_pair_prompt(a, b),
        max_retries=cfg.json_max_retries,
        max_tokens=320,
        temperature=cfg.temperature,
    )
    if not result.ok or not result.data:
        # A model that cannot answer must not silently produce "UNRELATED";
        # fall back to the rules, which at least justify themselves.
        return deterministic_relation(a, b), result.outcome

    verdict = result.data[0] if isinstance(result.data[0], dict) else {}
    relation = str(verdict.get("relation_type", "")).strip().upper()
    if relation not in VALID_RELATIONS:
        fallback = deterministic_relation(a, b)
        fallback["explanation"] = (
            f"Model returned an unrecognised relation "
            f"({verdict.get('relation_type')!r}); fell back to rule-based comparison. "
            + fallback["explanation"]
        )
        return fallback, result.outcome

    try:
        confidence = min(max(float(verdict.get("confidence", 0.6)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.6

    reason_tag = str(verdict.get("reason_tag") or "unspecified")[:60]
    explanation = str(verdict.get("explanation") or "").strip()[:1000]
    model_said = relation

    # The model reasonably but wrongly calls a period or unit mismatch a
    # contradiction. The deterministic check can prove it is not one, so it wins.
    if relation == "CONTRADICTS":
        blocked = _contradiction_blocker(a, b)
        if blocked is not None:
            reason_tag, blocker_note = blocked
            relation = "CONTEXT_RECONCILED"
            explanation = (
                f"{blocker_note} (Model called this {model_said}: "
                f"{explanation or 'no explanation given'})"
            )[:1000]

    # Symmetrically: same period, same magnitude is a corroboration the model
    # sometimes misses when the two facts are worded very differently.
    elif relation in ("CONTEXT_RECONCILED", "UNRELATED"):
        confirmed = _corroboration_signal(a, b)
        if confirmed is not None:
            relation = "CORROBORATES"
            reason_tag = "same_value"
            explanation = (
                f"{confirmed} (Model called this {model_said}: "
                f"{explanation or 'no explanation given'})"
            )[:1000]

    return (
        {
            "relation_type": relation,
            "reason_tag": reason_tag,
            "explanation": explanation,
            "confidence": confidence,
        },
        result.outcome,
    )


def store_relationship(
    conn: sqlite3.Connection,
    candidate: Candidate,
    verdict: dict,
    model_name: str,
    *,
    cross_document: bool = True,
) -> None:
    conn.execute(
        """INSERT INTO relationships
           (id, fact_a_id, fact_b_id, relation_type, reason_tag, explanation,
            confidence, similarity_score, cross_document, model, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(fact_a_id, fact_b_id) DO UPDATE SET
             relation_type=excluded.relation_type, reason_tag=excluded.reason_tag,
             explanation=excluded.explanation, confidence=excluded.confidence,
             similarity_score=excluded.similarity_score, model=excluded.model""",
        (
            f"rel_{candidate.fact_a_id[5:13]}_{candidate.fact_b_id[5:13]}",
            candidate.fact_a_id,
            candidate.fact_b_id,
            verdict["relation_type"],
            verdict.get("reason_tag"),
            verdict.get("explanation"),
            verdict.get("confidence", 0.5),
            candidate.similarity,
            1 if cross_document else 0,
            model_name,
            _now(),
        ),
    )


def link_facts(
    conn: sqlite3.Connection,
    cfg: Config,
    client: LLMClient | None,
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    on_progress=None,
) -> LinkStats:
    """Generate candidate pairs, classify them, and store the relationships."""
    import time

    started = time.perf_counter()
    stats = LinkStats()

    index = VectorIndex.load(conn)
    candidates = index.cross_document_pairs(
        top_k=top_k or cfg.candidate_top_k,
        threshold=threshold if threshold is not None else cfg.similarity_threshold,
    )
    if limit:
        candidates = candidates[:limit]
    stats.candidates = len(candidates)
    if not candidates:
        return stats

    existing: set[tuple[str, str]] = set()
    if skip_existing:
        existing = {
            (r["fact_a_id"], r["fact_b_id"])
            for r in conn.execute("SELECT fact_a_id, fact_b_id FROM relationships")
        }

    model_name = client.model_name if client else "deterministic"
    for index_i, candidate in enumerate(candidates, 1):
        if (candidate.fact_a_id, candidate.fact_b_id) in existing:
            stats.skipped_existing += 1
            if on_progress:
                on_progress(index_i, len(candidates), stats)
            continue

        a = load_fact_view(conn, candidate.fact_a_id)
        b = load_fact_view(conn, candidate.fact_b_id)
        if a is None or b is None:
            stats.failed += 1
            continue

        verdict, outcome = classify_pair(a, b, client, cfg)
        stats.repair_outcomes[outcome] += 1
        stats.classified += 1
        stats.by_type[verdict["relation_type"]] += 1
        stats.by_reason[verdict.get("reason_tag") or "unspecified"] += 1

        store_relationship(conn, candidate, verdict, model_name)
        stats.stored += 1

        conn.commit()
        if on_progress:
            on_progress(index_i, len(candidates), stats)

    conn.commit()
    with transaction(conn):
        conn.execute("UPDATE documents SET status='linked' WHERE status='extracted'")
    stats.elapsed_s = time.perf_counter() - started
    return stats
