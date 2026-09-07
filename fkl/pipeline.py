"""Pipeline orchestration.

Stage 1 implements ingestion (parse -> chunk -> store). Extraction and linking
hook in at the marked points in later stages.

Incremental ingestion
---------------------
A document's identity is the SHA-256 of its bytes. Consequences:

* Re-ingesting the same file is a no-op - nothing is reparsed, no facts are
  recomputed, and existing cross-document relationships stay valid.
* A modified file hashes differently and is treated as a genuinely new document,
  so the old version's facts remain intact and comparable against the new one.
* When `PARSER_VERSION` changes, previously-parsed documents are detected as
  stale and re-parsed automatically, without touching documents already at the
  current version.

Known limitation, stated honestly: this diffs at *document* granularity, not
page granularity. Re-ingesting a 100-page PDF that changed on one page redoes
all 100 pages. Page-level hashing would fix that and is noted in the README's
next steps.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import PARSER_VERSION, Config
from .db import transaction
from .ingest.chunker import chunk_page
from .ingest.pdf_parser import (
    document_id_from_hash,
    hash_file,
    is_probably_scanned,
    label_coverage,
    parse_pdf,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class IngestResult:
    document_id: str
    filename: str
    status: str  # ingested | skipped_unchanged | reparsed_stale_parser | error
    page_count: int = 0
    chunk_count: int = 0
    extraction_chunks: int = 0
    retrieval_chunks: int = 0
    table_chunks: int = 0
    label_coverage: float = 0.0
    probably_scanned: bool = False
    elapsed_s: float = 0.0
    message: str = ""

    @property
    def was_processed(self) -> bool:
        return self.status in {"ingested", "reparsed_stale_parser"}


def _existing_document(conn: sqlite3.Connection, content_sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE content_sha256 = ?", (content_sha256,)
    ).fetchone()


def ingest_pdf(
    conn: sqlite3.Connection,
    path: Path,
    cfg: Config,
    *,
    force: bool = False,
) -> IngestResult:
    """Parse and store one PDF. Safe to call repeatedly - see module docstring."""
    started = time.perf_counter()
    path = Path(path)
    if not path.exists():
        return IngestResult("", path.name, "error", message=f"file not found: {path}")

    # --- incremental check: hash first, parse only if we must -------------- #
    content_sha256 = hash_file(path)
    document_id = document_id_from_hash(content_sha256)
    existing = _existing_document(conn, content_sha256)

    if existing is not None and not force:
        if existing["parser_version"] == PARSER_VERSION:
            return IngestResult(
                document_id=existing["id"],
                filename=existing["filename"],
                status="skipped_unchanged",
                page_count=existing["page_count"] or 0,
                chunk_count=conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (existing["id"],)
                ).fetchone()[0],
                elapsed_s=time.perf_counter() - started,
                message="identical content already ingested",
            )
        status = "reparsed_stale_parser"
    else:
        status = "ingested"

    parsed = parse_pdf(path, cfg)

    # --- store ------------------------------------------------------------- #
    with transaction(conn):
        # Cascades away pages/chunks/facts/evidence for a re-parse.
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        coverage = label_coverage(parsed)
        scanned = is_probably_scanned(parsed)
        meta = {
            "printed_label_coverage": round(coverage, 4),
            "probably_scanned": scanned,
            "detected_tables": sum(len(p.tables) for p in parsed.pages),
        }
        conn.execute(
            """INSERT INTO documents
               (id, content_sha256, filename, source_path, title, page_count,
                byte_size, parser_version, status, ingested_at, meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                document_id,
                parsed.content_sha256,
                parsed.filename,
                parsed.source_path,
                parsed.title,
                parsed.page_count,
                parsed.byte_size,
                PARSER_VERSION,
                "parsed",
                _now(),
                json.dumps(meta),
            ),
        )

        extraction_ordinal = 0
        retrieval_ordinal = 0
        counts = {"extraction": 0, "retrieval": 0, "table": 0}

        for page in parsed.pages:
            page_id = f"{document_id}:p{page.pdf_page_index}"
            conn.execute(
                """INSERT INTO pages
                   (id, document_id, pdf_page_index, printed_page_label, text,
                    char_count, table_count)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    page_id,
                    document_id,
                    page.pdf_page_index,
                    page.printed_page_label,
                    page.text,
                    page.char_count,
                    len(page.tables),
                ),
            )

            for chunk in chunk_page(page, cfg):
                if chunk.role == "extraction":
                    ordinal = extraction_ordinal
                    extraction_ordinal += 1
                else:
                    ordinal = retrieval_ordinal
                    retrieval_ordinal += 1
                counts[chunk.role] += 1
                if chunk.kind == "table":
                    counts["table"] += 1

                chunk_id = f"{document_id}:{chunk.role[0]}{ordinal}"
                conn.execute(
                    """INSERT INTO chunks
                       (id, document_id, page_id, pdf_page_index, printed_page_label,
                        role, kind, ordinal, char_start, char_end, span_verified,
                        text, token_estimate, numeric_density)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chunk_id,
                        document_id,
                        page_id,
                        chunk.pdf_page_index,
                        chunk.printed_page_label,
                        chunk.role,
                        chunk.kind,
                        ordinal,
                        chunk.char_start,
                        chunk.char_end,
                        1 if chunk.span_verified else 0,
                        chunk.text,
                        chunk.token_estimate,
                        round(chunk.numeric_density, 4),
                    ),
                )

    return IngestResult(
        document_id=document_id,
        filename=parsed.filename,
        status=status,
        page_count=parsed.page_count,
        chunk_count=counts["extraction"] + counts["retrieval"],
        extraction_chunks=counts["extraction"],
        retrieval_chunks=counts["retrieval"],
        table_chunks=counts["table"],
        label_coverage=coverage,
        probably_scanned=scanned,
        elapsed_s=time.perf_counter() - started,
    )


def ingest_paths(
    conn: sqlite3.Connection,
    paths: list[Path],
    cfg: Config,
    *,
    force: bool = False,
    on_progress=None,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for i, path in enumerate(paths, 1):
        if on_progress:
            on_progress(i, len(paths), path)
        results.append(ingest_pdf(conn, path, cfg, force=force))
    return results


def discover_dataset(cfg: Config, dataset: str) -> list[Path]:
    """Resolve a dataset name to its PDFs.

    `dataset` is matched against subdirectory names of the datasets directory -
    there is no hardcoded list of dataset or file names anywhere, so dropping a
    new folder of PDFs in makes it selectable immediately.
    """
    root = cfg.datasets_dir
    if not root.exists():
        return []
    if dataset in {"all", "*"}:
        return sorted(p for p in root.glob("*/*.pdf"))
    matches = [d for d in sorted(root.iterdir()) if d.is_dir() and dataset.lower() in d.name.lower()]
    return sorted(p for d in matches for p in d.glob("*.pdf"))


def available_datasets(cfg: Config) -> list[str]:
    root = cfg.datasets_dir
    if not root.exists():
        return []
    return [d.name for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("*.pdf"))]
