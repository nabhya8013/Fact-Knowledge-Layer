"""Incremental-ingestion behaviour: identical bytes are a no-op, a parser-version
bump reparses, and deleting a document cascades everything derived from it away.

These run against a real starter PDF (the 27-page Q4 deck) so the hashing and the
cascade are exercised end to end, not mocked.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl import pipeline  # noqa: E402
from fkl.config import CONFIG  # noqa: E402
from fkl.db import connect, stats  # noqa: E402
from fkl.pipeline import ingest_pdf  # noqa: E402

PDF = (
    Path(__file__).resolve().parent.parent
    / "starter-datasets" / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf"
)


@pytest.fixture
def store(tmp_path):
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path)
    conn = connect(cfg.db_path)
    yield conn, cfg
    conn.close()


def test_identical_bytes_are_skipped_without_reparsing(store):
    conn, cfg = store
    first = ingest_pdf(conn, PDF, cfg)
    assert first.status == "ingested" and first.was_processed
    counts_before = stats(conn)

    second = ingest_pdf(conn, PDF, cfg)
    assert second.status == "skipped_unchanged"
    assert not second.was_processed
    assert second.document_id == first.document_id
    assert stats(conn) == counts_before  # nothing re-inserted


def test_parser_version_bump_marks_stale_and_reparses(store, monkeypatch):
    conn, cfg = store
    first = ingest_pdf(conn, PDF, cfg)
    page_ids_before = {
        r["id"] for r in conn.execute("SELECT id FROM pages WHERE document_id = ?", (first.document_id,))
    }

    monkeypatch.setattr(pipeline, "PARSER_VERSION", "999")
    again = ingest_pdf(conn, PDF, cfg)
    assert again.status == "reparsed_stale_parser"
    assert again.was_processed
    assert again.document_id == first.document_id  # same hash, same id

    row = conn.execute(
        "SELECT parser_version FROM documents WHERE id = ?", (first.document_id,)
    ).fetchone()
    assert row["parser_version"] == "999"
    # Old pages were replaced, not duplicated.
    assert conn.execute("SELECT COUNT(*) FROM pages WHERE document_id = ?",
                        (first.document_id,)).fetchone()[0] == len(page_ids_before)


def test_deleting_a_document_cascades(store):
    conn, cfg = store
    result = ingest_pdf(conn, PDF, cfg)
    assert stats(conn)["pages"] > 0 and stats(conn)["chunks"] > 0

    conn.execute("DELETE FROM documents WHERE id = ?", (result.document_id,))
    conn.commit()

    s = stats(conn)
    assert s["documents"] == 0
    assert s["pages"] == 0
    assert s["chunks"] == 0


def test_a_changed_file_becomes_a_new_document(store, tmp_path):
    conn, cfg = store
    ingest_pdf(conn, PDF, cfg)

    # One byte different -> different sha256 -> a genuinely new document.
    altered = tmp_path / "altered.pdf"
    altered.write_bytes(PDF.read_bytes() + b"\n")
    second = ingest_pdf(conn, altered, cfg)
    assert second.was_processed
    assert stats(conn)["documents"] == 2
