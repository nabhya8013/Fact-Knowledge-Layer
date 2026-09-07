"""Turn parsed pages into chunks, preserving exact character offsets.

Two chunk *roles* are produced from the same page text:

``extraction``
    Non-overlapping units handed to the LLM. Default granularity is a whole
    page, because a page of these documents is roughly 500-800 tokens - well
    inside context - and one LLM call per page is ~3x cheaper than one per
    sliding window. Pages that are too long are split at paragraph, then
    sentence, then hard boundaries.

``retrieval``
    Overlapping windows used *only* for embedding and similarity search.
    Overlap matters here (a fact split across a window boundary should still be
    findable) but would cause duplicate facts if used for extraction.

Every chunk records `char_start`/`char_end` into the page's verbatim text, so a
quoted span can always be resolved back to an exact location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Config
from .pdf_parser import ParsedPage

# Approximate tokens-per-character for English text. Only used to decide when to
# split; nothing depends on it being exact.
_CHARS_PER_TOKEN = 4.0

_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?:;])[ \t]*\n|(?<=[.!?])[ \t]+")
_LINE_RE = re.compile(r"\n")
_NUMBER_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    role: str  # extraction | retrieval
    kind: str  # prose | table
    text: str
    char_start: int | None
    char_end: int | None
    pdf_page_index: int
    printed_page_label: str | None
    span_verified: bool = True

    @property
    def token_estimate(self) -> int:
        return max(1, int(len(self.text) / _CHARS_PER_TOKEN))

    @property
    def numeric_density(self) -> float:
        """Digits per word. Used to skip units with no quantitative content."""
        words = len(_WORD_RE.findall(self.text))
        if not words:
            return 0.0
        return len(_NUMBER_RE.findall(self.text)) / words


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# --------------------------------------------------------------------------- #
# Offset-preserving splitting
# --------------------------------------------------------------------------- #


def _split_spans(text: str, pattern: re.Pattern[str], base: int = 0) -> list[tuple[int, int]]:
    """Split `text` on `pattern`, returning absolute (start, end) spans.

    Separators are consumed but the surviving spans keep their true offsets into
    the original string, which is the whole point: we must never lose the ability
    to map a chunk back to its exact source position.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in pattern.finditer(text):
        if m.start() > cursor:
            spans.append((base + cursor, base + m.start()))
        cursor = m.end()
    if cursor < len(text):
        spans.append((base + cursor, base + len(text)))
    return spans or [(base, base + len(text))]


def _pack_spans(
    text: str, spans: list[tuple[int, int]], max_chars: int, base: int
) -> list[tuple[int, int]]:
    """Greedily merge adjacent spans into groups no larger than `max_chars`."""
    out: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None

    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if end - cur_start <= max_chars:
            cur_end = end
        else:
            out.append((cur_start, cur_end))  # type: ignore[arg-type]
            cur_start, cur_end = start, end
    if cur_start is not None:
        out.append((cur_start, cur_end))  # type: ignore[arg-type]
    return out


def _hard_split(start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    return [(s, min(s + max_chars, end)) for s in range(start, end, max_chars)]


def split_text_spans(text: str, max_chars: int, base: int = 0) -> list[tuple[int, int]]:
    """Split into spans of at most `max_chars`, preferring natural boundaries.

    Escalates paragraph -> sentence -> line -> hard character split, only going
    to the next level for spans that are still too long.
    """
    if len(text) <= max_chars:
        return [(base, base + len(text))]

    spans = _pack_spans(text, _split_spans(text, _PARAGRAPH_RE, base), max_chars, base)

    for pattern in (_SENTENCE_RE, _LINE_RE):
        refined: list[tuple[int, int]] = []
        needs_more = False
        for start, end in spans:
            if end - start <= max_chars:
                refined.append((start, end))
                continue
            sub = text[start - base : end - base]
            pieces = _pack_spans(sub, _split_spans(sub, pattern, start), max_chars, start)
            refined.extend(pieces)
            needs_more = needs_more or any(e - s > max_chars for s, e in pieces)
        spans = refined
        if not needs_more:
            break

    final: list[tuple[int, int]] = []
    for start, end in spans:
        final.extend(_hard_split(start, end, max_chars) if end - start > max_chars else [(start, end)])
    return final


# --------------------------------------------------------------------------- #
# Chunk builders
# --------------------------------------------------------------------------- #


def build_extraction_chunks(page: ParsedPage, cfg: Config) -> list[Chunk]:
    """One unit per page, split further only when the page is too long.

    Tables that passed the parser's quality gate are emitted as *additional*
    chunks carrying the markdown rendering. They are marked `span_verified=False`
    when their span could not be located in the page text, because the markdown
    is a synthesised string rather than a substring of the page.
    """
    text = page.text
    chunks: list[Chunk] = []

    if text.strip():
        max_chars = int(cfg.max_extraction_tokens * _CHARS_PER_TOKEN)
        for start, end in split_text_spans(text, max_chars):
            piece = text[start:end]
            if len(piece.strip()) < cfg.min_chunk_chars:
                continue
            chunks.append(
                Chunk(
                    role="extraction",
                    kind="prose",
                    text=piece,
                    char_start=start,
                    char_end=end,
                    pdf_page_index=page.pdf_page_index,
                    printed_page_label=page.printed_page_label,
                )
            )

    for table in page.tables:
        if len(table.markdown.strip()) < cfg.min_chunk_chars:
            continue
        chunks.append(
            Chunk(
                role="extraction",
                kind="table",
                text=table.markdown,
                char_start=table.char_start,
                char_end=table.char_end,
                pdf_page_index=page.pdf_page_index,
                printed_page_label=page.printed_page_label,
                span_verified=table.char_start is not None,
            )
        )

    return chunks


def build_retrieval_chunks(page: ParsedPage, cfg: Config) -> list[Chunk]:
    """Overlapping windows over the page text, for embedding only."""
    text = page.text
    if not text.strip():
        return []

    size = cfg.retrieval_chunk_chars
    overlap = min(cfg.retrieval_chunk_overlap, max(0, size - 1))
    stride = max(1, size - overlap)

    chunks: list[Chunk] = []
    for start in range(0, max(1, len(text)), stride):
        end = min(start + size, len(text))
        piece = text[start:end]
        if len(piece.strip()) >= cfg.min_chunk_chars:
            chunks.append(
                Chunk(
                    role="retrieval",
                    kind="prose",
                    text=piece,
                    char_start=start,
                    char_end=end,
                    pdf_page_index=page.pdf_page_index,
                    printed_page_label=page.printed_page_label,
                )
            )
        if end >= len(text):
            break
    return chunks


def chunk_page(page: ParsedPage, cfg: Config) -> list[Chunk]:
    return build_extraction_chunks(page, cfg) + build_retrieval_chunks(page, cfg)
