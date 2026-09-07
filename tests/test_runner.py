"""Tests for the extraction runner: resume, worker sizing, and the parallel path.

The parallel path is worth a real test rather than a mocked one. It crosses a
process boundary with `spawn`, so the failure modes are unpicklable arguments,
workers that cannot import the package, and results that never come back - none
of which a single-process test would catch. Using LLM_BACKEND=none keeps it fast
while still exercising the actual multiprocessing machinery.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.config import CONFIG  # noqa: E402
from fkl.db import connect  # noqa: E402
from fkl.extract.extractor import mark_chunk_done, pending_chunks  # noqa: E402
from fkl.extract.runner import resolve_workers, run_extraction  # noqa: E402

# Quantitative sentences the deterministic extractor can find, quoted verbatim.
CHUNK_TEXTS = [
    "Revenue from operations stood at Rs 8,142 crore in FY24, up 12 per cent year on year. "
    "The company operated 42 distribution centres as of March 31, 2024.",
    "Real GDP growth is estimated at 6.4 per cent in FY25. "
    "Headline inflation moderated to 4.6 per cent during the same period.",
    "Total equity was 59,798.47 INR million at the end of the year. "
    "Borrowings stood at 1,005.28 INR million.",
    "The board met to discuss governance matters and long term strategy.",
]


@pytest.fixture
def store(tmp_path):
    """A small database with one document and a handful of extraction chunks."""
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path, llm_backend="none")
    conn = connect(cfg.db_path)

    doc_id = "doc_test0000000001"
    conn.execute(
        """INSERT INTO documents (id, content_sha256, filename, title, page_count,
                                  byte_size, parser_version, status, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_id, "0" * 64, "test.pdf", "Test Document", 1, 100, "1", "parsed", "2026-01-01"),
    )
    page_text = "\n\n".join(CHUNK_TEXTS)
    conn.execute(
        """INSERT INTO pages (id, document_id, pdf_page_index, printed_page_label,
                              text, char_count)
           VALUES (?,?,?,?,?,?)""",
        (f"{doc_id}:p0", doc_id, 0, "1", page_text, len(page_text)),
    )
    cursor = 0
    for i, text in enumerate(CHUNK_TEXTS):
        start = page_text.index(text, cursor)
        cursor = start + len(text)
        conn.execute(
            """INSERT INTO chunks (id, document_id, page_id, pdf_page_index,
                                   printed_page_label, role, kind, ordinal,
                                   char_start, char_end, text, token_estimate,
                                   numeric_density)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"{doc_id}:e{i}", doc_id, f"{doc_id}:p0", 0, "1", "extraction", "prose",
             i, start, cursor, text, len(text) // 4, 0.5),
        )
    conn.commit()
    yield conn, cfg, doc_id
    conn.close()


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_pending_excludes_chunks_already_done(store):
    conn, cfg, doc_id = store
    assert len(pending_chunks(conn, cfg)) == len(CHUNK_TEXTS)

    mark_chunk_done(conn, f"{doc_id}:e0", "done", 2)
    conn.commit()
    remaining = pending_chunks(conn, cfg)
    assert len(remaining) == len(CHUNK_TEXTS) - 1
    assert all(r["id"] != f"{doc_id}:e0" for r in remaining)


def test_failed_chunks_are_retried_on_a_later_run(store):
    """A chunk that failed should not be treated as finished."""
    conn, cfg, doc_id = store
    mark_chunk_done(conn, f"{doc_id}:e0", "failed", 0)
    conn.commit()
    assert any(r["id"] == f"{doc_id}:e0" for r in pending_chunks(conn, cfg))


def test_extraction_is_resumable(store):
    """Interrupt after one chunk, re-run, and the rest complete exactly once."""
    conn, cfg, _ = store

    first = run_extraction(conn, cfg, workers=1, limit=1)
    assert first.chunks_processed == 1
    facts_after_first = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    second = run_extraction(conn, cfg, workers=1)
    assert second.chunks_processed == len(CHUNK_TEXTS) - 1, "must not redo the first chunk"

    total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert total > facts_after_first
    assert conn.execute("SELECT COUNT(*) FROM extraction_progress").fetchone()[0] == len(CHUNK_TEXTS)

    # A third run has nothing left to do.
    assert run_extraction(conn, cfg, workers=1).chunks_processed == 0


def test_rerunning_does_not_duplicate_facts(store):
    conn, cfg, _ = store
    run_extraction(conn, cfg, workers=1)
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # Force re-processing of the same chunks without clearing facts.
    conn.execute("DELETE FROM extraction_progress")
    conn.commit()
    run_extraction(conn, cfg, workers=1)

    after = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert after == before, "content-derived fact ids must make re-extraction idempotent"


# --------------------------------------------------------------------------- #
# Storage invariants
# --------------------------------------------------------------------------- #


def test_every_stored_fact_has_grounded_evidence(store):
    """The system's central promise, asserted against the database."""
    conn, cfg, _ = store
    run_extraction(conn, cfg, workers=1)

    orphans = conn.execute(
        "SELECT COUNT(*) FROM facts f WHERE NOT EXISTS "
        "(SELECT 1 FROM evidence e WHERE e.fact_id = f.id)"
    ).fetchone()[0]
    assert orphans == 0

    rows = conn.execute(
        """SELECT e.quote, e.quote_char_start, e.quote_char_end, p.text AS page_text
             FROM evidence e JOIN pages p ON p.id = ?"""
        , (f"doc_test0000000001:p0",)
    ).fetchall()
    assert rows
    for row in rows:
        assert row["page_text"][row["quote_char_start"] : row["quote_char_end"]] == row["quote"]


def test_fact_types_registry_grows_from_the_documents(store):
    conn, cfg, _ = store
    run_extraction(conn, cfg, workers=1)
    types = conn.execute("SELECT COUNT(*) FROM fact_types").fetchone()[0]
    assert types > 0
    total = conn.execute("SELECT SUM(fact_count) FROM fact_types").fetchone()[0]
    assert total == conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


# --------------------------------------------------------------------------- #
# Parallel path - real processes, not mocks
# --------------------------------------------------------------------------- #


def test_parallel_path_really_uses_child_processes(store, monkeypatch):
    """Guard against the pool branch silently degrading to in-process work.

    With `spawn`, children re-import the module, so a monkeypatch applied in the
    parent cannot reach them. If `_run_one` is never called parent-side, the work
    genuinely crossed a process boundary. This matters because the branch is fast
    enough (~60ms for two workers, since the imports are light) that timing alone
    cannot distinguish parallel from serial.

    Patching `_run_one` itself is not an option: `Pool` pickles that function, so
    replacing it with a test double breaks the very path under test. Spying on
    the pool construction works instead, because that happens only in the parent.
    """
    import multiprocessing as real_mp

    from fkl.extract import runner

    contexts: list[str] = []
    original_get_context = real_mp.get_context

    def spy(method=None):
        contexts.append(method)
        return original_get_context(method)

    monkeypatch.setattr(runner.mp, "get_context", spy)

    conn, cfg, _ = store
    result = run_extraction(conn, cfg, workers=2)

    assert result.chunks_processed == len(CHUNK_TEXTS)
    assert contexts == ["spawn"], (
        "the pool branch was bypassed, or used a start method other than spawn "
        "(fork is unsafe once a model or GPU context is loaded)"
    )


def test_serial_path_runs_in_process(store, monkeypatch):
    """The mirror image, so the assertion above cannot pass for the wrong reason."""
    from fkl.extract import runner

    parent_calls: list[int] = []
    original = runner._run_one
    monkeypatch.setattr(
        runner, "_run_one", lambda job: (parent_calls.append(1), original(job))[1]
    )

    conn, cfg, _ = store
    run_extraction(conn, cfg, workers=1)
    assert len(parent_calls) == len(CHUNK_TEXTS)


def test_parallel_extraction_produces_the_same_facts_as_serial(tmp_path):
    """Crossing a process boundary must not change the result."""
    serial_cfg = dataclasses.replace(CONFIG, data_dir=tmp_path / "serial", llm_backend="none")
    parallel_cfg = dataclasses.replace(CONFIG, data_dir=tmp_path / "parallel", llm_backend="none")

    results = {}
    for label, cfg in (("serial", serial_cfg), ("parallel", parallel_cfg)):
        conn = connect(cfg.db_path)
        doc_id = "doc_test0000000001"
        conn.execute(
            """INSERT INTO documents (id, content_sha256, filename, title, page_count,
                                      byte_size, parser_version, status, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (doc_id, "0" * 64, "t.pdf", "T", 1, 100, "1", "parsed", "2026-01-01"),
        )
        page_text = "\n\n".join(CHUNK_TEXTS)
        conn.execute(
            "INSERT INTO pages (id, document_id, pdf_page_index, text, char_count) VALUES (?,?,?,?,?)",
            (f"{doc_id}:p0", doc_id, 0, page_text, len(page_text)),
        )
        cursor = 0
        for i, text in enumerate(CHUNK_TEXTS):
            start = page_text.index(text, cursor)
            cursor = start + len(text)
            conn.execute(
                """INSERT INTO chunks (id, document_id, page_id, pdf_page_index, role, kind,
                                       ordinal, char_start, char_end, text, token_estimate,
                                       numeric_density)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"{doc_id}:e{i}", doc_id, f"{doc_id}:p0", 0, "extraction", "prose",
                 i, start, cursor, text, len(text) // 4, 0.5),
            )
        conn.commit()

        run_extraction(conn, cfg, workers=1 if label == "serial" else 2)
        results[label] = sorted(
            r[0] for r in conn.execute("SELECT id FROM facts").fetchall()
        )
        conn.close()

    assert results["serial"], "the serial run should have produced facts"
    assert results["serial"] == results["parallel"]


# --------------------------------------------------------------------------- #
# Worker sizing
# --------------------------------------------------------------------------- #


def test_explicit_worker_count_is_respected():
    assert resolve_workers(CONFIG, 3) == 3


def test_deterministic_backend_uses_one_worker():
    assert resolve_workers(dataclasses.replace(CONFIG, llm_backend="none")) == 1


def test_worker_count_is_bounded_and_positive():
    workers = resolve_workers(dataclasses.replace(CONFIG, llm_backend="local"))
    assert 1 <= workers <= 4
