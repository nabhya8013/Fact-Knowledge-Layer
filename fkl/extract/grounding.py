"""Verify that a quoted span really occurs in the source text, and locate it.

Why this module exists
----------------------
Every fact must carry an evidence quote that can be resolved to exact character
offsets in the page text. Naive `page_text.find(quote)` fails almost always on
real PDFs, because PyMuPDF hard-wraps lines mid-sentence:

    "...increased by 17.9 per cent in value terms and 35 \\nper cent..."

An LLM asked to copy that sentence verbatim will render the line break as a
space. Rejecting it as "not found" would throw away nearly every true quote and
leave only the accidentally-unwrapped ones - a silent, catastrophic recall bug.

So matching runs in tiers, from strictest to most forgiving, and records which
tier succeeded. Crucially, *every* tier returns offsets into the ORIGINAL text,
so the stored evidence still points at the real span, character-exact. A quote
that matches no tier is treated as ungrounded and its fact is discarded, which
is the main defence against a small model inventing numbers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters that PDFs use decoratively and models silently substitute.
_CHAR_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
    "­": "",       # soft hyphen
    "​": "",       # zero-width space
    "﻿": "",       # BOM
}

_MIN_QUOTE_CHARS = 12


@dataclass(frozen=True)
class QuoteMatch:
    """A located quote. `start`/`end` always index the ORIGINAL text."""

    start: int
    end: int
    text: str  # the original substring, i.e. source_text[start:end]
    match_mode: str  # exact | normalised_whitespace | normalised_chars | fuzzy_prefix


def _fold_chars(text: str) -> str:
    """Map typographic variants to ASCII equivalents, one-for-one where possible.

    Length is preserved except for characters mapped to "" or "..."; the index
    map built alongside handles that, so offsets stay correct either way.
    """
    return "".join(_CHAR_FOLD.get(ch, ch) for ch in text)


def _normalise(text: str, *, fold_chars: bool, casefold: bool) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space, returning an index map.

    `index_map[i]` is the offset in `text` of the character that produced
    `normalised[i]`. That is what lets a match found in normalised space be
    reported as exact offsets in the original.
    """
    out: list[str] = []
    index_map: list[int] = []
    prev_space = True  # suppresses leading whitespace

    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                index_map.append(i)
                prev_space = True
            continue
        replacement = _CHAR_FOLD.get(ch, ch) if fold_chars else ch
        if replacement == "":
            continue
        if casefold:
            replacement = replacement.lower()
        for sub in replacement:
            out.append(sub)
            index_map.append(i)
        prev_space = False

    return "".join(out), index_map


def _normalise_query(text: str, *, fold_chars: bool, casefold: bool) -> str:
    if fold_chars:
        text = _fold_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if casefold else text


def find_quote(source_text: str, quote: str, *, min_chars: int = _MIN_QUOTE_CHARS) -> QuoteMatch | None:
    """Locate `quote` within `source_text`, tolerating cosmetic differences.

    Returns exact original offsets, or None if the quote cannot be grounded.
    """
    if not quote or not source_text:
        return None

    cleaned = quote.strip()
    # Models often wrap the quote in their own quotation marks.
    if len(cleaned) >= 2 and cleaned[0] in "\"'“‘" and cleaned[-1] in "\"'”’":
        cleaned = cleaned[1:-1].strip()

    if len(cleaned) < min_chars:
        # Too short to be meaningful evidence; also too likely to match by chance.
        return None

    # Tier 1: the quote is a genuine substring.
    idx = source_text.find(cleaned)
    if idx >= 0:
        return QuoteMatch(idx, idx + len(cleaned), source_text[idx : idx + len(cleaned)], "exact")

    # Tiers 2-3: forgive line wrapping, then typography and case.
    for mode, fold_chars, casefold in (
        ("normalised_whitespace", False, False),
        ("normalised_chars", True, True),
    ):
        norm_text, index_map = _normalise(source_text, fold_chars=fold_chars, casefold=casefold)
        norm_query = _normalise_query(cleaned, fold_chars=fold_chars, casefold=casefold)
        if not norm_query:
            continue
        pos = norm_text.find(norm_query)
        if pos < 0:
            continue
        start = index_map[pos]
        end = index_map[pos + len(norm_query) - 1] + 1
        return QuoteMatch(start, end, source_text[start:end], mode)

    return None


def verify_quote(source_text: str, quote: str) -> QuoteMatch | None:
    """Alias with the intent spelled out: a fact survives only if this returns."""
    return find_quote(source_text, quote)


def locate_in_page(
    page_text: str, chunk_text: str, chunk_char_start: int | None, quote: str
) -> QuoteMatch | None:
    """Find a quote, preferring the chunk the model actually saw.

    Searching the chunk first prevents a quote from being attributed to a
    coincidentally identical string elsewhere on the page. Offsets are then
    translated into page coordinates. Falls back to a whole-page search, which
    matters for table chunks whose markdown is synthesised rather than a
    substring of the page.
    """
    match = find_quote(chunk_text, quote)
    if match is not None and chunk_char_start is not None:
        return QuoteMatch(
            match.start + chunk_char_start,
            match.end + chunk_char_start,
            match.text,
            match.match_mode,
        )
    if match is not None:
        # Chunk text is synthesised (e.g. table markdown): no page offset exists.
        return match
    return find_quote(page_text, quote)
