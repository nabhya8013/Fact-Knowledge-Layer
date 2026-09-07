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


_TOKEN_RE = re.compile(r"[0-9][0-9,\.]*|[A-Za-z]{4,}")
_NUMERIC_TOKEN_RE = re.compile(r"^[0-9][0-9,\.]*$")

# A reconstructed span may not sprawl across a whole page; if the tokens are not
# close together, the "fact" is an assembly of unrelated cells.
_MAX_RECONSTRUCTED_SPAN = 240
_MIN_WORD_COVERAGE = 0.75
# A single fact cites one figure or a small handful (a value and its prior-year
# comparative). A quote carrying many distinct numbers is a chart caption or a
# table region the model dumped wholesale, not one fact - refuse to ground it.
_MAX_RECONSTRUCTED_NUMERALS = 4


def find_reconstructed_span(
    source_text: str, quote: str, *, max_span: int = _MAX_RECONSTRUCTED_SPAN
) -> QuoteMatch | None:
    """Ground a quote that the model assembled from a table rather than copied.

    PyMuPDF emits table text column-major, so a row label, its value and the
    column's unit header are not contiguous in the page text. Asked for a
    verbatim quote, the model reasonably writes "Total equity 59,798.47 INR
    million" - every piece of which is on the page, in three different places.
    Rejecting that loses most facts on financial-statement pages, which is where
    the densest facts live.

    So instead of matching the model's string, we locate the smallest window of
    REAL source text containing all of its anchor tokens, and store that window
    as the evidence. The model's reconstruction is never stored as if it were a
    quote. Guard rails keep this from degenerating into "it appears somewhere on
    the page":

    * every numeric token must be present - numbers are the fact's substance, so
      a hallucinated figure still cannot be grounded;
    * most substantive words must be present;
    * the window must be short, so unrelated cells cannot be stitched together.
    """
    tokens = _TOKEN_RE.findall(quote)
    if not tokens:
        return None

    numeric = [t for t in tokens if _NUMERIC_TOKEN_RE.match(t)]
    words = [t.lower() for t in tokens if not _NUMERIC_TOKEN_RE.match(t)]

    # Numbers are mandatory, not merely preferred. This tier exists for
    # column-major numeric tables, where the figures are what pin the span down.
    # Allowing word-only matches made it grab whatever happened to be nearby: on
    # a prospectus cover page, "registered office" grounded to a 200-character
    # blob spanning six unrelated headers. For prose facts the model can and
    # should quote verbatim, so those belong to the stricter tiers above.
    if not numeric:
        return None

    # Too many distinct figures means the model quoted a chart caption or a whole
    # table region, not a single fact. A mangled Q4-deck caption once grounded a
    # "revenue" fact to the wrong number this way. Reject it - the fact is not
    # cleanly locatable and a wrong figure is worse than a missing one.
    if len(set(numeric)) > _MAX_RECONSTRUCTED_NUMERALS:
        return None

    haystack = source_text.lower()

    # Anchor on the rarest numeric token: fewest candidate positions to test.
    anchors = numeric or words
    anchor = min(anchors, key=lambda t: haystack.count(t.lower()) or 10**6)
    anchor_lower = anchor.lower()
    if anchor_lower not in haystack:
        return None

    best: tuple[int, int, float] | None = None
    start_search = 0
    while (pos := haystack.find(anchor_lower, start_search)) >= 0:
        start_search = pos + 1
        lo = max(0, pos - max_span)
        hi = min(len(source_text), pos + max_span)
        window = haystack[lo:hi]

        # Every number must be here, or this is not the right row.
        if any(n.lower() not in window for n in numeric):
            continue
        present = [w for w in words if w in window]
        coverage = len(present) / len(words) if words else 1.0
        if coverage < _MIN_WORD_COVERAGE:
            continue

        # Tighten to the smallest span actually covering the matched tokens.
        needles = [n.lower() for n in numeric] + present
        positions = []
        for needle in needles:
            idx = window.find(needle)
            if idx >= 0:
                positions.append((idx, idx + len(needle)))
        if not positions:
            continue
        span_start = lo + min(p[0] for p in positions)
        span_end = lo + max(p[1] for p in positions)
        if span_end - span_start > max_span:
            continue

        if best is None or coverage > best[2]:
            best = (span_start, span_end, coverage)

    if best is None:
        return None
    start, end, _ = best
    return QuoteMatch(start, end, source_text[start:end], "reconstructed_span")


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
    match = find_quote(chunk_text, quote) or find_reconstructed_span(chunk_text, quote)
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
