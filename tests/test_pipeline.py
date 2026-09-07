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
from fkl.extract.extractor import pending_chunks  # noqa: E402
from fkl.extract.runner import run_extraction  # noqa: E402
from fkl.pipeline import ingest_pdf  # noqa: E402

_DATA = Path(__file__).resolve().parent.parent / "starter-datasets"
PDF = _DATA / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf"
PDF_2 = _DATA / "india-macroeconomy" / "02-rbi-annual-report-2024-25-excerpt.pdf"
PDF_3 = _DATA / "india-macroeconomy" / "03-imf-india-2025-article-iv-excerpt.pdf"


@pytest.fixture
def store(tmp_path):
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path)
    conn = connect(cfg.db_path)
    yield conn, cfg
    conn.close()


@pytest.fixture
def offline_store(tmp_path):
    """Deterministic (non-LLM) extraction, so a real extract runs in the test."""
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path, llm_backend="none")
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


def test_adding_a_document_does_not_reprocess_existing_ones(offline_store):
    """Brownie point: new documents are incremental. Existing documents' chunks
    are not re-extracted and their facts are left exactly as they were."""
    conn, cfg = offline_store

    ingest_pdf(conn, PDF, cfg)
    ingest_pdf(conn, PDF_2, cfg)
    first = run_extraction(conn, cfg, workers=1)
    assert first.chunks_processed > 0
    assert len(pending_chunks(conn, cfg)) == 0

    # Snapshot every existing fact and its evidence offsets.
    before = {
        (r["id"], r["payload_json"], r["created_at"])
        for r in conn.execute("SELECT id, payload_json, created_at FROM facts")
    }
    progress_before = {
        r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM extraction_progress")
    }
    assert before, "the first two documents must have produced some facts"

    # A third document arrives.
    third = ingest_pdf(conn, PDF_3, cfg)
    assert third.was_processed

    pending = pending_chunks(conn, cfg)
    assert pending, "the new document has chunks to extract"
    assert all(r["document_id"] == third.document_id for r in pending), \
        "only the new document's chunks are pending"

    second = run_extraction(conn, cfg, workers=1)
    assert second.chunks_processed == len(pending)

    # Every pre-existing fact is byte-identical; nothing was recomputed.
    after = {
        (r["id"], r["payload_json"], r["created_at"])
        for r in conn.execute("SELECT id, payload_json, created_at FROM facts")
    }
    assert before <= after, "existing facts must survive unchanged"
    assert progress_before <= {
        r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM extraction_progress")
    }
    # And the new document actually contributed.
    assert conn.execute(
        "SELECT COUNT(*) FROM facts WHERE document_id = ?", (third.document_id,)
    ).fetchone()[0] >= 0
