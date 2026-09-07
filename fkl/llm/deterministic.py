"""Deterministic, non-LLM extraction and relation classification.

This is the floor the system never falls through. If the model download fails,
if a machine cannot run llama.cpp, or if someone sets `LLM_BACKEND=none`, the
pipeline still ingests documents, still produces grounded facts with real
evidence, and still links them across documents - just with lower recall and
blunter reasoning.

It is honest about what it is. Facts produced here are stored with
`extractor='deterministic'` so they are distinguishable in the UI and in any
evaluation, and it makes no attempt to imitate the LLM's judgement: relations
are decided by explicit arithmetic and scope rules, which is exactly why it can
never hallucinate one.
"""

from __future__ import annotations

import re

from ..extract.normalize import (
    CURRENCIES,
    SCALES,
    canonical_key,
    normalise_period,
    normalize_value,
    values_agree,
)

# A sentence must contain a number AND a quantity marker to be worth extracting.
_QUANTITY_HINT = re.compile(
    r"(\d)\s*(%|per\s*cent|percent|bps)"
    r"|([₹$€£¥])\s*[\d]"
    r"|(?<![a-z])(" + "|".join(sorted(SCALES, key=len, reverse=True)) + r")(?![a-z])"
    r"|(?<![a-z])(" + "|".join(re.escape(c) for c in CURRENCIES if c.isalpha()) + r")(?![a-z])",
    re.IGNORECASE,
)
_NUMBER_IN_TEXT = re.compile(r"\d")

# Sentence segmentation that keeps absolute offsets.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n{2,}")

_ATTRIBUTE_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "or", "is",
    "was", "were", "are", "be", "been", "by", "with", "from", "as", "that",
    "this", "it", "its", "our", "we", "which", "has", "had", "have", "up",
    "down", "about", "over", "under", "during", "stood", "reported", "recorded",
    "rose", "grew", "increased", "decreased", "declined", "reached", "posted",
}

_PERIOD_RE = re.compile(
    r"(FY\s?\d{2,4}(?:-\d{2,4})?|Q[1-4]\s?FY\s?\d{2,4}|"
    r"\d{4}-\d{2,4}|"
    r"(?:as (?:of|at)|ended|ending)\s+\d{1,2}\s+\w+\s+\d{4}|"
    r"(?:as (?:of|at)|in|during|for)\s+(?:FY\s?)?\d{4})",
    re.IGNORECASE,
)

_CAPITALISED_RUN = re.compile(r"\b([A-Z][A-Za-z&.']*(?:\s+[A-Z][A-Za-z&.']*){0,3})\b")


def _sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[cursor : m.start()]
        if piece.strip():
            spans.append((cursor, m.start(), piece))
        cursor = m.end()
    if cursor < len(text) and text[cursor:].strip():
        spans.append((cursor, len(text), text[cursor:]))
    return spans


def _guess_attribute(sentence: str, number_pos: int) -> str:
    """Words immediately before the number, minus filler, as the attribute."""
    before = sentence[:number_pos]
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", before)
    kept: list[str] = []
    for word in reversed(words):
        if word.lower() in _ATTRIBUTE_STOP:
            if kept:
                break
            continue
        kept.append(word)
        if len(kept) >= 4:
            break
    return " ".join(reversed(kept)).strip() or "value"


def _guess_subject(sentence: str, fallback: str | None) -> str:
    for match in _CAPITALISED_RUN.finditer(sentence):
        candidate = match.group(1).strip()
        # Skip sentence-initial single words, which are usually just the first
        # word capitalised rather than a real entity.
        if len(candidate.split()) >= 2 or match.start() > 0:
            if candidate.lower() not in _ATTRIBUTE_STOP:
                return candidate
    return fallback or "unspecified"


def deterministic_facts(
    text: str, *, document_title: str | None = None, max_facts: int = 12
) -> list[dict]:
    """Extract quantitative facts using patterns only. No model involved."""
    facts: list[dict] = []
    for _, _, sentence in _sentences_with_offsets(text):
        stripped = sentence.strip()
        if len(stripped) < 25 or len(stripped) > 600:
            continue
        if not _NUMBER_IN_TEXT.search(stripped) or not _QUANTITY_HINT.search(stripped):
            continue

        number_match = re.search(r"[\d][\d,\.]*", stripped)
        if not number_match:
            continue

        value = number_match.group(0).rstrip(".,")
        tail = stripped[number_match.end() : number_match.end() + 24]
        unit_match = re.match(
            r"\s*(%|per\s*cent|percent|bps|"
            + "|".join(sorted(SCALES, key=len, reverse=True))
            + r")",
            tail,
            re.IGNORECASE,
        )
        unit = unit_match.group(1) if unit_match else None
        # Pull a currency symbol appearing just before the number into the unit.
        prefix = stripped[max(0, number_match.start() - 12) : number_match.start()]
        currency = next((c for c in "₹$€£¥" if c in prefix), None)
        if currency:
            unit = f"{currency} {unit}".strip() if unit else currency

        period = _PERIOD_RE.search(stripped)

        facts.append(
            {
                "subject": _guess_subject(stripped, document_title),
                "attribute": _guess_attribute(stripped, number_match.start()),
                "value": value,
                "unit": unit,
                "time_scope": period.group(1) if period else None,
                "qualifier": None,
                "confidence": 0.35,  # deliberately low: this is pattern matching
                "source_quote": stripped,
            }
        )
        if len(facts) >= max_facts:
            break
    return facts


def deterministic_relation(fact_a: dict, fact_b: dict) -> dict:
    """Classify a pair by arithmetic and scope rules rather than judgement.

    Cannot hallucinate a relationship, but also cannot recognise that two
    differently-named attributes mean the same thing - which is precisely the
    gap the LLM classifier fills.
    """
    key_a = canonical_key(fact_a.get("subject"), fact_a.get("attribute"))
    key_b = canonical_key(fact_b.get("subject"), fact_b.get("attribute"))

    value_a, unit_a = normalize_value(fact_a.get("value"), fact_a.get("unit"))
    value_b, unit_b = normalize_value(fact_b.get("value"), fact_b.get("unit"))

    scope_a = normalise_period(fact_a.get("time_scope"))
    scope_b = normalise_period(fact_b.get("time_scope"))
    qual_a = (fact_a.get("qualifier") or "").strip().lower()
    qual_b = (fact_b.get("qualifier") or "").strip().lower()

    if key_a != key_b:
        return {
            "relation_type": "UNRELATED",
            "reason_tag": "different_attribute",
            "explanation": f"Different canonical keys: '{key_a}' versus '{key_b}'.",
            "confidence": 0.5,
        }

    if unit_a and unit_b and unit_a != unit_b:
        return {
            "relation_type": "CONTEXT_RECONCILED",
            "reason_tag": "units_differ",
            "explanation": (
                f"Same property but incomparable units: {unit_a} versus {unit_b}. "
                "No conversion is applied."
            ),
            "confidence": 0.5,
        }

    agree = values_agree(value_a, value_b)
    if agree is None:
        return {
            "relation_type": "UNRELATED",
            "reason_tag": "non_numeric",
            "explanation": "At least one value is not numeric, so no comparison is possible.",
            "confidence": 0.3,
        }

    if agree:
        return {
            "relation_type": "CORROBORATES",
            "reason_tag": "same_value",
            "explanation": (
                f"Same property and matching magnitude ({value_a:g} {unit_a or ''}"
                f" versus {value_b:g} {unit_b or ''})."
            ),
            "confidence": 0.7,
        }

    if scope_a != scope_b:
        return {
            "relation_type": "CONTEXT_RECONCILED",
            "reason_tag": "different_period",
            "explanation": (
                f"Values differ ({value_a:g} versus {value_b:g}) but the periods differ too: "
                f"'{fact_a.get('time_scope')}' versus '{fact_b.get('time_scope')}'."
            ),
            "confidence": 0.6,
        }

    if qual_a != qual_b:
        return {
            "relation_type": "CONTEXT_RECONCILED",
            "reason_tag": "different_scope",
            "explanation": (
                f"Values differ ({value_a:g} versus {value_b:g}) but the stated scope differs: "
                f"'{fact_a.get('qualifier')}' versus '{fact_b.get('qualifier')}'."
            ),
            "confidence": 0.6,
        }

    return {
        "relation_type": "CONTRADICTS",
        "reason_tag": "value_mismatch",
        "explanation": (
            f"Same property, same period and same scope, but the values differ: "
            f"{value_a:g} versus {value_b:g}."
        ),
        "confidence": 0.6,
    }
