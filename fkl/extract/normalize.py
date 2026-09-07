"""Normalise fact values, units and identities so facts become comparable.

The linking stage has to decide whether "Rs 8,142 crore" and "INR 81.42 billion"
are the same quantity. That is impossible on raw strings, so every numeric fact
is reduced to a magnitude in a *base unit of its own currency/dimension*:

    "Rs 8,142 crore"  -> value_num = 8.142e10, unit = "INR"
    "INR 81.42 bn"    -> value_num = 8.142e10, unit = "INR"   (corroborates)
    "$1.2 billion"    -> value_num = 1.2e9,    unit = "USD"   (not comparable to INR)
    "17.9 per cent"   -> value_num = 17.9,     unit = "%"

Cross-currency conversion is deliberately NOT attempted - that would need an FX
rate and a date, and inventing one would manufacture false corroborations. Facts
in different currencies stay incomparable numerically and are left to the LLM
classifier to reason about.

Indian-convention digit grouping ("12,34,567") is handled by simply stripping
separators, which works for both Indian and Western grouping.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Scale words
# --------------------------------------------------------------------------- #

SCALES: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3, "k": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "tr": 1e12,
}

CURRENCIES: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR",
    "rupee": "INR", "rupees": "INR",
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "¥": "JPY", "jpy": "JPY",
}

# Dimensionless / ratio units, normalised to a canonical spelling.
RATIO_UNITS: dict[str, str] = {
    "%": "%", "percent": "%", "per cent": "%", "pct": "%", "percentage": "%",
    "percentage point": "pp", "percentage points": "pp", "pp": "pp",
    "bps": "bps", "basis point": "bps", "basis points": "bps",
    "x": "x", "times": "x",
}

# Order matters: the separator-grouped alternative must require at least one
# separator (`+`, not `*`). With `*` it matches just the leading 1-3 digits and,
# because regex alternation is ordered, wins before the plain-number branch is
# ever tried - silently turning "8142" into 814 and "2024" into 202.
_NUMBER_RE = re.compile(
    r"""(?P<sign>[-+−]?)\s*
        (?P<num>
            \d{1,3}(?:[,\s]\d{2,3})+(?:\.\d+)?   # 1,234,567 or 12,34,567
          | \d+(?:\.\d+)?                        # 8142 or 81.42
          | \.\d+                                # .5
        )""",
    re.VERBOSE,
)
_PAREN_NEGATIVE_RE = re.compile(r"^\s*\(\s*(.+?)\s*\)\s*$")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def parse_number(text: str) -> float | None:
    """Extract the first numeric magnitude from a string, or None.

    Handles thousands separators in either convention, unicode minus, and the
    accounting convention where parentheses denote a negative.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    cleaned = _clean(text)
    if not cleaned:
        return None

    negative = False
    if m := _PAREN_NEGATIVE_RE.match(cleaned):
        negative = True
        cleaned = m.group(1)

    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None

    digits = re.sub(r"[, \s]", "", match.group("num"))
    try:
        value = float(digits)
    except ValueError:
        return None

    if match.group("sign") in {"-", "−"}:
        negative = True
    return -value if negative else value


def detect_scale(text: str) -> float:
    """Multiplier implied by a scale word anywhere in the string (default 1.0)."""
    lowered = _clean(text).lower()
    best = 1.0
    for word, mult in SCALES.items():
        # Word-boundary match so "million" does not fire inside "millionaire"
        # and the single-letter aliases do not fire inside ordinary words.
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
            best = max(best, mult)
    return best


def detect_currency(*texts: str) -> str | None:
    for text in texts:
        if not text:
            continue
        lowered = _clean(text).lower()
        for token, code in CURRENCIES.items():
            if token in "₹$€£¥":
                if token in lowered:
                    return code
            elif re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lowered):
                return code
    return None


def detect_ratio_unit(*texts: str) -> str | None:
    for text in texts:
        if not text:
            continue
        lowered = _clean(text).lower()
        if "%" in lowered:
            return "%"
        for token, code in sorted(RATIO_UNITS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lowered):
                return code
    return None


def normalize_value(value, unit: str | None = None) -> tuple[float | None, str | None]:
    """Reduce (value, unit) to a magnitude plus a canonical unit.

    Returns (None, canonical_unit) when the value carries no parseable number,
    which is normal for qualitative facts ("market leader in express parcel").
    """
    value_text = "" if value is None else str(value)
    unit_text = unit or ""
    combined = f"{value_text} {unit_text}"

    number = parse_number(value_text)
    if number is None:
        number = parse_number(unit_text)

    ratio = detect_ratio_unit(value_text, unit_text)
    if ratio is not None:
        return (number, ratio)

    currency = detect_currency(value_text, unit_text)
    scale = max(detect_scale(value_text), detect_scale(unit_text))

    if number is None:
        return (None, currency or (_clean(unit_text) or None))

    magnitude = number * scale
    if currency:
        return (magnitude, currency)

    # A bare scaled count, e.g. "12.5 million shipments".
    residual = _clean(unit_text).lower()
    for word in SCALES:
        residual = re.sub(rf"(?<![a-z]){re.escape(word)}(?![a-z])", "", residual)
    residual = re.sub(r"\s+", " ", residual).strip(" .,")
    return (magnitude, residual or None)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "s",
    "total", "overall",
}


def slugify(text: str) -> str:
    """Lowercase, strip punctuation and stopwords, collapse to a stable token."""
    cleaned = _clean(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s%]", " ", cleaned)
    words = [w for w in cleaned.split() if w and w not in _STOPWORDS]
    return "_".join(words)


def canonical_key(subject: str | None, attribute: str | None) -> str:
    """Grouping key for facts describing the same property of the same entity."""
    return f"{slugify(subject or 'unknown')}::{slugify(attribute or 'unknown')}"


def canonical_text(payload: dict) -> str:
    """The string that gets embedded for similarity search.

    Built from the fact's *fields* rather than the raw sentence, so that two
    documents phrasing the same fact very differently still land near each other
    in embedding space - which is the whole point of embedding facts rather than
    source text.
    """
    subject = _clean(str(payload.get("subject") or ""))
    attribute = _clean(str(payload.get("attribute") or payload.get("predicate") or ""))
    value = _clean(str(payload.get("value") if payload.get("value") is not None else ""))
    unit = _clean(str(payload.get("unit") or ""))
    scope = _clean(str(payload.get("time_scope") or payload.get("period") or ""))

    parts = [p for p in (subject, attribute) if p]
    head = " - ".join(parts) if parts else "unknown fact"
    tail = " ".join(p for p in (value, unit) if p)
    if tail:
        head = f"{head}: {tail}"
    if scope:
        head = f"{head} [{scope}]"
    return head


def fact_type_name(payload: dict) -> str:
    """The dynamic-schema registry key: the normalised attribute name.

    New attribute strings create new fact types with no migration, which is how
    the schema grows to fit whatever the documents happen to contain.
    """
    attribute = payload.get("attribute") or payload.get("predicate") or "unspecified"
    return slugify(str(attribute)) or "unspecified"


def values_agree(a: float | None, b: float | None, tolerance: float = 0.01) -> bool | None:
    """Whether two magnitudes agree within a relative tolerance.

    Returns None when the comparison is not defined (either side missing), so
    callers can distinguish "disagrees" from "cannot tell" instead of silently
    treating unknown as false.
    """
    if a is None or b is None:
        return None
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= tolerance
