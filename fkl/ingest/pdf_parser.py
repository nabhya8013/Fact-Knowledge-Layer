"""PDF -> pages, with everything needed to ground a fact back to its source.

Two things make grounding possible later:

1. `ParsedPage.text` is the *verbatim* string PyMuPDF returned. We never strip,
   normalise or re-wrap it, because every character offset stored anywhere else
   indexes into this exact string.
2. `printed_page_label` records the page number *printed on the page*, which in
   curated excerpts is not the physical page index. Citing "page 221" when the
   reader's PDF viewer says "page 50" would be useless, so we keep both.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from ..config import Config

# A line consisting of nothing but a page number, optionally decorated.
_PAGE_LABEL_RE = re.compile(
    r"^[\s\|\-–—\.]*(?:page\s+)?(\d{1,4}|[ivxlcdm]{1,7})[\s\|\-–—\.]*$",
    re.IGNORECASE,
)
# A running header/footer that *ends* or *begins* with the page number, e.g.
# "INTERNATIONAL MONETARY FUND 37" or "46  ANNUAL REPORT 2024-25". Restricted to
# short lines so that body text and data rows cannot qualify.
_EDGE_NUMBER_RE = re.compile(r"^(?:(\d{1,4})\s+\D.*|.*\D\s+(\d{1,4}))$")
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# How many lines at the top / bottom of a page may hold the page number. Real
# documents put a few lines of running furniture (section name, company name)
# between the page number and the page edge, so this needs slack.
_LABEL_SCAN_LINES = 8
# Maximum length of a line that may carry a page number alongside other text.
_EDGE_LINE_MAX_CHARS = 60


@dataclass
class ParsedTable:
    """A table PyMuPDF detected and that passed the shape quality gate."""

    markdown: str
    row_count: int
    col_count: int
    char_start: int | None  # span in the page text this table covers, if locatable
    char_end: int | None


@dataclass
class ParsedPage:
    pdf_page_index: int  # 0-based physical page
    text: str  # verbatim; all offsets index into this
    printed_page_label: str | None = None
    tables: list[ParsedTable] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ParsedDocument:
    content_sha256: str
    filename: str
    source_path: str
    title: str | None
    page_count: int
    byte_size: int
    pages: list[ParsedPage]


# --------------------------------------------------------------------------- #
# Page-label detection
# --------------------------------------------------------------------------- #


def _roman_to_int(value: str) -> int | None:
    value = value.lower()
    total = 0
    prev = 0
    for ch in reversed(value):
        digit = _ROMAN_VALUES.get(ch)
        if digit is None:
            return None
        total = total - digit if digit < prev else total + digit
        prev = max(prev, digit)
    return total or None


def _label_candidates(text: str) -> list[tuple[str, int]]:
    """Page-number candidates from the head and tail of a page, as (label, value).

    Two shapes are recognised, both common in real documents:

    * a line that is *only* the page number ("38", "xiv");
    * a short running header/footer that starts or ends with it
      ("INTERNATIONAL MONETARY FUND 37", "46  ANNUAL REPORT 2024-25").

    Documents frequently alternate between the two on facing pages, so
    supporting only the first shape halves coverage and then breaks the
    neighbour-agreement check that validates candidates.
    """
    lines = text.splitlines()
    if not lines:
        return []
    scan: list[str] = lines[:_LABEL_SCAN_LINES] + lines[-_LABEL_SCAN_LINES:]

    out: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        value = _roman_to_int(token) if _ROMAN_RE.match(token) else int(token)
        if value is None or value <= 0 or value > 9999:
            return
        if token not in seen:
            seen.add(token)
            out.append((token, value))

    for line in scan:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 12 and (m := _PAGE_LABEL_RE.match(stripped)):
            add(m.group(1))
        elif len(stripped) <= _EDGE_LINE_MAX_CHARS and (m := _EDGE_NUMBER_RE.match(stripped)):
            add(m.group(1) or m.group(2))
    return out


def _longest_consecutive_run(values: list[int]) -> list[int]:
    """Longest run of consecutive integers in a sorted, de-duplicated list."""
    best: list[int] = []
    current: list[int] = []
    for v in values:
        if current and v == current[-1] + 1:
            current.append(v)
        else:
            current = [v]
        if len(current) > len(best):
            best = list(current)
    return best


def _resolve_page_labels(pages: Iterable[ParsedPage]) -> None:
    """Assign `printed_page_label` using agreement between neighbouring pages.

    A number is only accepted as a page label if an adjacent page carries the
    number immediately before or after it. That is deliberately conservative: it
    ignores stray figures ("Table 4", years, bare data cells) and it survives the
    page-number jumps that curated excerpts introduce, without any per-document
    rule. Pages whose candidates find no neighbour support keep `None`.

    A single PDF page can legitimately carry *several* printed page numbers: some
    reports are typeset as 2-up spreads, so physical page 40 is printed pages
    80 and 81. When that happens the surviving candidates form a consecutive run
    and we record the range ("80-81") rather than silently picking one half.
    """
    page_list = list(pages)
    candidates = [_label_candidates(p.text) for p in page_list]

    for i, page in enumerate(page_list):
        # Keep every candidate that a neighbouring page corroborates. For a
        # 2-up spread the neighbour is two printed pages away, so accept a
        # match at the neighbour's own offset as well.
        supported: dict[int, str] = {}
        for label, value in candidates[i]:
            for offset in (-1, 1):
                j = i + offset
                if not (0 <= j < len(page_list)):
                    continue
                neighbour_values = {v for _, v in candidates[j]}
                if any(value + offset * step in neighbour_values for step in (1, 2)):
                    supported[value] = label
                    break

        if not supported:
            page.printed_page_label = None
            continue

        run = _longest_consecutive_run(sorted(supported))
        if len(run) > 1:
            page.printed_page_label = f"{supported[run[0]]}-{supported[run[-1]]}"
        else:
            page.printed_page_label = supported[run[0]]


# --------------------------------------------------------------------------- #
# Table detection
# --------------------------------------------------------------------------- #


def _locate_span(haystack: str, needles: Iterable[str]) -> tuple[int | None, int | None]:
    """Smallest span of `haystack` covering the given cell strings, if findable."""
    starts, ends = [], []
    for needle in needles:
        needle = needle.strip()
        if len(needle) < 3:
            continue
        idx = haystack.find(needle)
        if idx >= 0:
            starts.append(idx)
            ends.append(idx + len(needle))
    if not starts:
        return None, None
    return min(starts), max(ends)


_NUMERIC_TOKEN_RE = re.compile(r"^[\(\-–]?[\d,]+(?:\.\d+)?[\)%]?$")


def _looks_tabular(page_text: str, min_numeric_lines: int) -> bool:
    """Cheap pre-check: does this page plausibly contain a table at all?

    PyMuPDF's table detector costs ~0.1-2s per page depending on density, which
    dominates ingestion when run over every page of a 100-page financial report.
    Nearly all of that work is wasted on pure-prose pages. This test costs
    microseconds and only has to be *permissive* - a false positive merely means
    we pay for the real detector, while the shape gate below still decides what
    is kept.

    The signal is generic: a run of lines whose tokens are predominantly
    numeric. No document, publisher or layout is special-cased.
    """
    numeric_lines = 0
    for line in page_text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        hits = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t))
        # A lone number on its own line (common in column-major PDF text output)
        # counts, as does a line that is mostly numbers.
        if hits and hits >= max(1, len(tokens) // 2):
            numeric_lines += 1
            if numeric_lines >= min_numeric_lines:
                return True
    return False


def _rows_to_markdown(rows: list[list]) -> str:
    """Render extracted cells as a markdown table.

    Built here rather than via `Table.to_markdown()` because that method re-runs
    the (expensive) cell extraction we have already paid for.
    """
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""

    def cell(value) -> str:
        return str(value if value is not None else "").replace("\n", " ").replace("|", "\\|").strip()

    lines = ["| " + " | ".join(cell(c) for c in (rows[0] + [""] * (width - len(rows[0])))) + " |"]
    lines.append("|" + "---|" * width)
    for row in rows[1:]:
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(cell(c) for c in padded) + " |")
    return "\n".join(lines)


def _extract_tables(page: "fitz.Page", page_text: str, cfg: Config) -> list[ParsedTable]:
    """Detect tables, keeping only those with a plausible grid shape.

    PyMuPDF's detector does poorly on the borderless statistical tables common in
    financial and institutional PDFs: it tends to either collapse the whole table
    into a single cell or shatter column boundaries. Rather than special-case any
    document, we apply a generic shape gate - a kept table must have at least
    `table_min_rows` rows and `table_min_cols` columns. Anything failing the gate
    is dropped and the ordinary page text is used instead, which for these files
    is usually the more faithful representation anyway.
    """
    if not cfg.enable_table_extraction:
        return []
    if not _looks_tabular(page_text, cfg.table_probe_min_numeric_lines):
        return []

    try:
        finder = page.find_tables(strategy=cfg.table_strategy)
    except Exception:
        return []

    kept: list[ParsedTable] = []
    for table in getattr(finder, "tables", []):
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows:
            continue
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        if n_rows < cfg.table_min_rows or n_cols < cfg.table_min_cols:
            continue
        markdown = _rows_to_markdown(rows)
        if not markdown.strip():
            continue

        cells = [str(c) for row in rows for c in row if c]
        start, end = _locate_span(page_text, cells)
        kept.append(
            ParsedTable(
                markdown=markdown,
                row_count=n_rows,
                col_count=n_cols,
                char_start=start,
                char_end=end,
            )
        )
    return kept


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def hash_file(path: Path) -> str:
    """SHA-256 of the file bytes. This is the document identity used for
    incremental ingestion: re-uploading identical bytes is a no-op."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdf(path: Path, cfg: Config) -> ParsedDocument:
    """Parse a PDF into pages with verbatim text, page labels and tables."""
    path = Path(path)
    content_sha256 = hash_file(path)

    with fitz.open(str(path)) as doc:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or None

        pages: list[ParsedPage] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            # "text" mode preserves reading order and is what our offsets index.
            text = page.get_text("text")
            parsed = ParsedPage(pdf_page_index=index, text=text)
            parsed.tables = _extract_tables(page, text, cfg)
            pages.append(parsed)

        page_count = doc.page_count

    _resolve_page_labels(pages)

    # Fall back to the PDF's own title metadata, then the filename stem.
    if not title:
        title = path.stem.replace("-", " ").replace("_", " ").strip()

    return ParsedDocument(
        content_sha256=content_sha256,
        filename=path.name,
        source_path=str(path),
        title=title,
        page_count=page_count,
        byte_size=path.stat().st_size,
        pages=pages,
    )


def document_id_from_hash(content_sha256: str) -> str:
    return f"doc_{content_sha256[:16]}"


def is_probably_scanned(parsed: ParsedDocument, min_chars_per_page: float = 120.0) -> bool:
    """Rough OCR check. We do not run OCR, but we want to *report* when a PDF
    yields almost no text so the failure is visible rather than silent."""
    if not parsed.pages:
        return True
    total = sum(p.char_count for p in parsed.pages)
    return (total / len(parsed.pages)) < min_chars_per_page


def label_coverage(parsed: ParsedDocument) -> float:
    """Fraction of pages for which a printed page label was resolved."""
    if not parsed.pages:
        return 0.0
    found = sum(1 for p in parsed.pages if p.printed_page_label)
    return found / len(parsed.pages)


def label_histogram(parsed: ParsedDocument) -> Counter:
    return Counter(p.printed_page_label is not None for p in parsed.pages)
