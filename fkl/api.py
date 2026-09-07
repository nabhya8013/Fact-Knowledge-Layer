"""FastAPI application: upload PDFs, inspect facts, browse relationships.

Connection handling
-------------------
Every request opens its own SQLite connection and closes it. Connections are not
shareable across threads, and FastAPI runs sync endpoints in a threadpool, so a
module-level connection would be a latent corruption bug rather than an
optimisation. Opening a SQLite connection is microseconds; WAL means readers
never block on the background writer.

Background work
---------------
Ingestion, extraction and linking all take far longer than an HTTP request. Each
is dispatched to a worker thread and tracked in the `jobs` table, so the UI can
poll for progress and the browser is never left hanging.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import CONFIG
from .db import connect, json_load, stats as db_stats
from .showcase import _fact_detail, build_showcase, failure_report

WEB_DIR = CONFIG.project_root / "web"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    return connect(CONFIG.db_path)


app = FastAPI(
    title="Fact Knowledge Layer",
    description="Extract grounded facts from PDFs and link them across documents.",
    version="0.1.0",
)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


def _set_job(job_id: str, **fields) -> None:
    conn = get_conn()
    try:
        columns = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*fields.values(), job_id))
        conn.commit()
    finally:
        conn.close()


def _create_job(kind: str, document_id: str | None = None) -> str:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO jobs (id, document_id, kind, status, progress, message, started_at)
               VALUES (?,?,?,'queued',0,?,?)""",
            (job_id, document_id, kind, "queued", _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def _run_job(job_id: str, fn) -> None:
    """Run `fn(report)` on a worker thread, recording success or failure."""

    def report(progress: float, message: str) -> None:
        _set_job(job_id, progress=progress, message=message[:300])

    def target() -> None:
        _set_job(job_id, status="running", message="starting")
        try:
            summary = fn(report)
            _set_job(
                job_id, status="done", progress=1.0,
                message=str(summary)[:300], finished_at=_now(),
            )
        except Exception as exc:  # noqa: BLE001 - surface the error to the UI
            _set_job(
                job_id, status="error",
                message=f"{exc.__class__.__name__}: {exc}"[:300],
                finished_at=_now(),
            )
            traceback.print_exc()

    threading.Thread(target=target, daemon=True).start()


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict[str, Any]:
    from .llm.factory import describe_backend

    conn = get_conn()
    try:
        counts = db_stats(conn)
    finally:
        conn.close()
    return {
        "status": "ok",
        "llm_backend": describe_backend(CONFIG),
        "embedding_model": CONFIG.embedding_model,
        "similarity_threshold": CONFIG.similarity_threshold,
        "counts": counts,
    }


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT d.*,
                      (SELECT COUNT(*) FROM facts f WHERE f.document_id = d.id) AS fact_count,
                      (SELECT COUNT(*) FROM chunks c
                        WHERE c.document_id = d.id AND c.role='extraction') AS chunk_count,
                      (SELECT COUNT(*) FROM extraction_progress ep
                         JOIN chunks c2 ON c2.id = ep.chunk_id
                        WHERE c2.document_id = d.id) AS chunks_done
                 FROM documents d ORDER BY d.ingested_at DESC"""
        ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["meta"] = json_load(r["meta_json"], {})
            item.pop("meta_json", None)
            out.append(item)
        return out
    finally:
        conn.close()


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, f"no such document: {document_id}")
        return {"deleted": document_id}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


@app.get("/api/facts")
def list_facts(
    document_id: str | None = None,
    fact_type: str | None = None,
    q: str | None = Query(None, description="substring match on the fact text"),
    numeric_only: bool = False,
    min_confidence: float = 0.0,
    limit: int = Query(50, le=500),
    offset: int = 0,
) -> dict[str, Any]:
    conn = get_conn()
    try:
        where = ["f.confidence >= ?"]
        params: list[Any] = [min_confidence]
        if document_id:
            where.append("f.document_id = ?")
            params.append(document_id)
        if fact_type:
            where.append("f.fact_type = ?")
            params.append(fact_type)
        if numeric_only:
            where.append("f.value_num IS NOT NULL")
        if q:
            where.append("(f.canonical_text LIKE ? OR f.payload_json LIKE ?)")
            params += [f"%{q}%", f"%{q}%"]
        clause = " AND ".join(where)

        total = conn.execute(f"SELECT COUNT(*) FROM facts f WHERE {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT f.id FROM facts f WHERE {clause}
                ORDER BY f.confidence DESC, f.id LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        items = [_fact_detail(conn, r["id"]) for r in rows]
        return {"total": total, "limit": limit, "offset": offset,
                "items": [i for i in items if i]}
    finally:
        conn.close()


@app.get("/api/facts/{fact_id}")
def get_fact(fact_id: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        detail = _fact_detail(conn, fact_id)
        if detail is None:
            raise HTTPException(404, f"no such fact: {fact_id}")

        related = conn.execute(
            """SELECT r.*, CASE WHEN r.fact_a_id = ? THEN r.fact_b_id ELSE r.fact_a_id END AS other_id
                 FROM relationships r
                WHERE r.fact_a_id = ? OR r.fact_b_id = ?
                ORDER BY r.confidence DESC""",
            (fact_id, fact_id, fact_id),
        ).fetchall()

        detail["relationships"] = [
            {
                "relation_type": r["relation_type"],
                "reason_tag": r["reason_tag"],
                "explanation": r["explanation"],
                "confidence": r["confidence"],
                "similarity_score": r["similarity_score"],
                "other_fact": _fact_detail(conn, r["other_id"]),
            }
            for r in related
        ]
        return detail
    finally:
        conn.close()


@app.get("/api/fact-types")
def list_fact_types() -> list[dict[str, Any]]:
    """The dynamic schema registry - what the documents turned out to contain."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT ft.*, d.filename AS first_seen_document
                 FROM fact_types ft
                 LEFT JOIN documents d ON d.id = ft.first_seen_document_id
                ORDER BY ft.fact_count DESC"""
        ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["observed_keys"] = json_load(r["observed_keys_json"], [])
            item.pop("observed_keys_json", None)
            out.append(item)
        return out
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Relationships and showcase
# --------------------------------------------------------------------------- #


@app.get("/api/relationships")
def list_relationships(
    relation_type: str | None = None,
    reason_tag: str | None = None,
    include_unrelated: bool = False,
    limit: int = Query(50, le=500),
    offset: int = 0,
) -> dict[str, Any]:
    conn = get_conn()
    try:
        where, params = ["1=1"], []
        if relation_type:
            where.append("relation_type = ?")
            params.append(relation_type.upper())
        elif not include_unrelated:
            where.append("relation_type != 'UNRELATED'")
        if reason_tag:
            where.append("reason_tag = ?")
            params.append(reason_tag)
        clause = " AND ".join(where)

        total = conn.execute(
            f"SELECT COUNT(*) FROM relationships WHERE {clause}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""SELECT * FROM relationships WHERE {clause}
                ORDER BY confidence DESC, similarity_score DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()

        items = []
        for r in rows:
            items.append(
                {
                    "id": r["id"],
                    "relation_type": r["relation_type"],
                    "reason_tag": r["reason_tag"],
                    "explanation": r["explanation"],
                    "confidence": r["confidence"],
                    "similarity_score": r["similarity_score"],
                    "model": r["model"],
                    "fact_a": _fact_detail(conn, r["fact_a_id"]),
                    "fact_b": _fact_detail(conn, r["fact_b_id"]),
                }
            )
        by_type = dict(
            conn.execute(
                "SELECT relation_type, COUNT(*) FROM relationships GROUP BY relation_type"
            ).fetchall()
        )
        return {"total": total, "limit": limit, "offset": offset,
                "by_type": by_type, "items": items}
    finally:
        conn.close()


@app.get("/api/showcase")
def showcase() -> dict[str, Any]:
    """The four required demonstration cases, in one place."""
    conn = get_conn()
    try:
        return build_showcase(conn)
    finally:
        conn.close()


@app.get("/api/failures")
def failures() -> dict[str, Any]:
    conn = get_conn()
    try:
        return failure_report(conn, sample_size=10)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


@app.get("/api/evidence/{fact_id}/context")
def evidence_context(fact_id: str, window: int = Query(400, le=4000)) -> dict[str, Any]:
    """The quote shown inside its surrounding page text, for verification.

    This is what lets a reviewer confirm a citation without opening the PDF: the
    quote is returned with its real offsets and the text on either side of it.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            """SELECT e.*, p.text AS page_text, d.filename
                 FROM evidence e
                 JOIN chunks c ON c.id = e.chunk_id
                 JOIN pages p ON p.id = c.page_id
                 JOIN documents d ON d.id = e.document_id
                WHERE e.fact_id = ? LIMIT 1""",
            (fact_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"no evidence for fact: {fact_id}")

        page_text = row["page_text"]
        start, end = row["quote_char_start"], row["quote_char_end"]
        if start is None or end is None:
            return {"fact_id": fact_id, "quote": row["quote"], "context": None,
                    "filename": row["filename"]}

        lo, hi = max(0, start - window), min(len(page_text), end + window)
        return {
            "fact_id": fact_id,
            "filename": row["filename"],
            "pdf_page_index": row["pdf_page_index"],
            "printed_page_label": row["printed_page_label"],
            "match_mode": row["match_mode"],
            "quote": row["quote"],
            "char_start": start,
            "char_end": end,
            "before": page_text[lo:start],
            "highlight": page_text[start:end],
            "after": page_text[end:hi],
            # Proof the stored offsets still address the stored quote.
            "offsets_verified": page_text[start:end] == row["quote"],
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Pipeline actions
# --------------------------------------------------------------------------- #


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), autorun: bool = True) -> dict[str, Any]:
    """Accept a PDF, ingest it, then optionally extract and link in background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")

    CONFIG.ensure_dirs()
    target = CONFIG.uploads_dir / file.filename
    with open(target, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    from .pipeline import ingest_pdf

    conn = get_conn()
    try:
        result = ingest_pdf(conn, target, CONFIG)
    finally:
        conn.close()

    if result.status == "error":
        raise HTTPException(400, result.message)

    payload = {
        "document_id": result.document_id,
        "filename": result.filename,
        "status": result.status,
        "page_count": result.page_count,
        "extraction_chunks": result.extraction_chunks,
        "printed_label_coverage": result.label_coverage,
        "probably_scanned": result.probably_scanned,
        "job_id": None,
    }

    if autorun and result.was_processed:
        job_id = _create_job("extract_and_link", result.document_id)
        payload["job_id"] = job_id

        def work(report):
            from .extract.runner import run_extraction
            from .link.relate import link_facts
            from .link.vector_index import index_facts
            from .llm.factory import build_client

            job_conn = get_conn()
            client = None
            try:
                client = build_client(CONFIG, quiet=True)

                def progress(done, total, _stats):
                    report(0.8 * done / max(1, total), f"extracting {done}/{total} chunks")

                ex = run_extraction(
                    job_conn, CONFIG, document_id=result.document_id,
                    on_progress=progress,
                )
                report(0.85, "embedding facts")
                index_facts(job_conn, CONFIG)
                report(0.9, "linking across documents")
                ln = link_facts(job_conn, CONFIG, client)
                return (f"{ex.facts_stored} facts, {ln.stored} relationships")
            finally:
                if client:
                    client.close()
                job_conn.close()

        _run_job(job_id, work)

    return payload


@app.post("/api/extract")
def start_extraction(document_id: str | None = None) -> dict[str, Any]:
    job_id = _create_job("extract", document_id)

    def work(report):
        from .extract.runner import run_extraction

        conn = get_conn()
        try:
            def progress(done, total, _stats):
                report(done / max(1, total), f"{done}/{total} chunks")

            stats = run_extraction(conn, CONFIG, document_id=document_id, on_progress=progress)
            return f"{stats.facts_stored} facts from {stats.chunks_processed} chunks"
        finally:
            conn.close()

    _run_job(job_id, work)
    return {"job_id": job_id}


@app.post("/api/link")
def start_linking(threshold: float | None = None) -> dict[str, Any]:
    job_id = _create_job("link")

    def work(report):
        from .link.relate import link_facts
        from .link.vector_index import index_facts
        from .llm.factory import build_client

        conn = get_conn()
        client = None
        try:
            report(0.1, "embedding facts")
            index_facts(conn, CONFIG)
            client = build_client(CONFIG, quiet=True)

            def progress(done, total, _stats):
                report(0.1 + 0.9 * done / max(1, total), f"{done}/{total} pairs")

            stats = link_facts(conn, CONFIG, client, threshold=threshold, on_progress=progress)
            return f"{stats.stored} relationships from {stats.candidates} candidates"
        finally:
            if client:
                client.close()
            conn.close()

    _run_job(job_id, work)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no such job: {job_id}")
        return dict(row)
    finally:
        conn.close()


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Static UI (mounted last so it cannot shadow /api routes)
# --------------------------------------------------------------------------- #


@app.get("/")
def index() -> Any:
    target = WEB_DIR / "index.html"
    if not target.exists():
        return JSONResponse(
            {"error": "web/index.html not found", "api_docs": "/docs"}, status_code=404
        )
    return FileResponse(target)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
