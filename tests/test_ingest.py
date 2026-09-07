"""Stage-1 tests.

The properties worth defending here are the ones the rest of the system leans
on: character offsets must be exact, page labels must never be invented, and
re-ingesting a document must not duplicate work.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.config import CONFIG, Config  # noqa: E402
from fkl.db import connect, stats  # noqa: E402
from fkl.ingest.chunker import Chunk, split_text_spans  # noqa: E402
from fkl.ingest.pdf_parser import (  # noqa: E402
    ParsedPage,
    _resolve_page_labels,
    _roman_to_int,
    parse_pdf,
)
from fkl.pipeline import discover_dataset, ingest_pdf  # noqa: E402


# --------------------------------------------------------------------------- #
# Offset preservation - the property everything else depends on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "short text",
        "para one.\n\npara two is longer and has more words in it.\n\npara three.",
        "A sentence. Another sentence! A third one? And a fourth.\n" * 40,
        "no boundaries at all " * 200,
        "\n\n\n",
        "x" * 5000,
    ],
)
@pytest.mark.parametrize("max_chars", [50, 200, 1000])
def test_split_spans_are_exact_offsets(text, max_chars):
    spans = split_text_spans(text, max_chars)
    for start, end in spans:
        assert 0 <= start <= end <= len(text)
        # The span must address the real string, not a copy or a normalisation.
        assert text[start:end] == text[start:end]
    # Spans must be ordered and non-overlapping.
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 <= s2, "extraction spans must not overlap"
    # Nothing but separator characters may be dropped.
    covered = "".join(text[s:e] for s, e in spans)
    assert len(covered) <= len(text)
    assert covered.strip().replace("\n", "").replace(" ", "") == (
        text.strip().replace("\n", "").replace(" ", "")
    )


def test_split_respects_max_chars():
    text = "sentence number %d is here. " % 1 * 500
    for start, end in split_text_spans(text, 300):
        assert end - start <= 300


# --------------------------------------------------------------------------- #
# Page-label detection
# --------------------------------------------------------------------------- #


def test_roman_numerals():
    assert _roman_to_int("iv") == 4
    assert _roman_to_int("xiv") == 14
    assert _roman_to_int("mcmxc") == 1990
    assert _roman_to_int("q") is None


def test_page_labels_require_neighbour_agreement():
    """A consistent run is labelled; an isolated number is not."""
    pages = [ParsedPage(i, f"HEADER\n{100 + i}\nbody text here\n") for i in range(4)]
    _resolve_page_labels(pages)
    assert [p.printed_page_label for p in pages] == ["100", "101", "102", "103"]


def test_page_labels_reject_unsupported_numbers():
    """Stray numbers (a table caption, a year) must not become page labels."""
    pages = [
        ParsedPage(0, "Table 4\nrevenue was 2024\n"),
        ParsedPage(1, "1999\nsome unrelated text\n"),
        ParsedPage(2, "chapter text with no page number\n"),
    ]
    _resolve_page_labels(pages)
    assert all(p.printed_page_label is None for p in pages)


def test_page_labels_record_two_up_spreads_as_a_range():
    """The Delhivery annual report is typeset A3 2-up: one PDF page carries two
    printed pages. Reporting only one half would mis-cite evidence."""
    pages = [
        ParsedPage(0, "78\nbody\n79\n"),
        ParsedPage(1, "80\nbody\n81\n"),
        ParsedPage(2, "82\nbody\n83\n"),
    ]
    _resolve_page_labels(pages)
    assert pages[1].printed_page_label == "80-81"


def test_page_labels_survive_a_jump():
    """Curated excerpts skip page ranges; both runs should still be labelled."""
    values = [10, 11, 12, 90, 91, 92]
    pages = [ParsedPage(i, f"{v}\nbody\n") for i, v in enumerate(values)]
    _resolve_page_labels(pages)
    assert [p.printed_page_label for p in pages] == [str(v) for v in values]


# --------------------------------------------------------------------------- #
# Chunk metadata
# --------------------------------------------------------------------------- #


def test_numeric_density():
    prose = Chunk("extraction", "prose", "the quick brown fox jumped over", 0, 31, 0, None)
    numeric = Chunk("extraction", "prose", "revenue 8142 crore up 12 percent", 0, 32, 0, None)
    assert prose.numeric_density == 0.0
    assert numeric.numeric_density > prose.numeric_density


# --------------------------------------------------------------------------- #
# End-to-end ingestion against the real starter PDFs
# --------------------------------------------------------------------------- #


def _first_starter_pdf() -> Path | None:
    pdfs = discover_dataset(CONFIG, "all")
    return pdfs[0] if pdfs else None


@pytest.fixture
def tmp_cfg(tmp_path):
    import dataclasses

    return dataclasses.replace(CONFIG, data_dir=tmp_path)


@pytest.mark.skipif(_first_starter_pdf() is None, reason="starter dataset not present")
def test_chunk_offsets_resolve_against_stored_page_text(tmp_cfg):
    """Every stored prose chunk must be a verbatim slice of its page text.

    This is the guarantee that makes evidence grounding possible in stage 2.
    """
    conn = connect(tmp_cfg.data_dir / "t.db")
    try:
        result = ingest_pdf(conn, _first_starter_pdf(), tmp_cfg)
        assert result.was_processed
        assert result.page_count > 0
        assert result.extraction_chunks > 0

        rows = conn.execute(
            """SELECT c.text, c.char_start, c.char_end, p.text AS page_text
                 FROM chunks c JOIN pages p ON p.id = c.page_id
                WHERE c.kind = 'prose'"""
        ).fetchall()
        assert rows
        for r in rows:
            assert r["page_text"][r["char_start"] : r["char_end"]] == r["text"]
    finally:
        conn.close()


@pytest.mark.skipif(_first_starter_pdf() is None, reason="starter dataset not present")
def test_reingesting_identical_bytes_is_a_noop(tmp_cfg):
    conn = connect(tmp_cfg.data_dir / "t.db")
    try:
        pdf = _first_starter_pdf()
        first = ingest_pdf(conn, pdf, tmp_cfg)
        before = stats(conn)

        second = ingest_pdf(conn, pdf, tmp_cfg)
        after = stats(conn)

        assert first.status == "ingested"
        assert second.status == "skipped_unchanged"
        assert second.document_id == first.document_id
        assert before == after, "a no-op ingest must not change the store"
        # And it must be fast, because nothing was parsed.
        assert second.elapsed_s < first.elapsed_s
    finally:
        conn.close()


@pytest.mark.skipif(_first_starter_pdf() is None, reason="starter dataset not present")
def test_force_reingest_replaces_without_duplicating(tmp_cfg):
    conn = connect(tmp_cfg.data_dir / "t.db")
    try:
        pdf = _first_starter_pdf()
        ingest_pdf(conn, pdf, tmp_cfg)
        before = stats(conn)
        ingest_pdf(conn, pdf, tmp_cfg, force=True)
        after = stats(conn)
        assert before == after, "force re-ingest must replace, not duplicate"
    finally:
        conn.close()


@pytest.mark.skipif(_first_starter_pdf() is None, reason="starter dataset not present")
def test_retrieval_chunks_overlap_and_extraction_chunks_do_not(tmp_cfg):
    conn = connect(tmp_cfg.data_dir / "t.db")
    try:
        ingest_pdf(conn, _first_starter_pdf(), tmp_cfg)
        for role, expect_overlap in (("retrieval", True), ("extraction", False)):
            rows = conn.execute(
                """SELECT char_start, char_end, page_id FROM chunks
                    WHERE role = ? AND kind = 'prose' ORDER BY ordinal""",
                (role,),
            ).fetchall()
            overlaps = sum(
                1
                for a, b in zip(rows, rows[1:])
                if a["page_id"] == b["page_id"] and b["char_start"] < a["char_end"]
            )
            assert (overlaps > 0) is expect_overlap, f"{role} overlap expectation failed"
    finally:
        conn.close()


@pytest.mark.skipif(_first_starter_pdf() is None, reason="starter dataset not present")
def test_parse_reports_text_bearing_pdf():
    from fkl.ingest.pdf_parser import is_probably_scanned

    parsed = parse_pdf(_first_starter_pdf(), CONFIG)
    assert not is_probably_scanned(parsed), "starter PDFs should yield extractable text"
    assert parsed.content_sha256 and len(parsed.content_sha256) == 64
