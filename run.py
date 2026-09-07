#!/usr/bin/env python3
"""Fact Knowledge Layer - single entry point.

    python run.py                     # preflight + ingest the default dataset
    python run.py ingest --dataset all
    python run.py ingest path/to/a.pdf path/to/b.pdf
    python run.py status
    python run.py doc <document_id>
    python run.py page <document_id> <pdf_page_index>
    python run.py chunks <document_id> [--role extraction] [--limit 5]
    python run.py reset

No accounts, no API keys and no background services are required.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
TESTED_MAX_PYTHON = (3, 13)

# Modules needed for the stage-1 (ingestion) commands only. Later stages add
# their own checks so that ingestion still works if, say, the model download
# failed.
_INGEST_REQUIREMENTS = [("fitz", "pymupdf")]


def preflight(require: list[tuple[str, str]] | None = None) -> None:
    """Fail loudly and usefully rather than with a stack trace 200 lines in."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(found {sys.version.split()[0]}).\n"
            "Install a newer Python, e.g.  uv python install 3.12"
        )
    if sys.version_info[:2] > TESTED_MAX_PYTHON:
        print(
            f"[preflight] note: Python {sys.version_info.major}.{sys.version_info.minor} is newer "
            f"than the tested {TESTED_MAX_PYTHON[0]}.{TESTED_MAX_PYTHON[1]}. "
            "Everything should still work; report anything that does not.",
            file=sys.stderr,
        )

    missing: list[str] = []
    for module, package in require or _INGEST_REQUIREMENTS:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        sys.exit(
            "Missing dependencies: " + ", ".join(missing) + "\n"
            "Install them with:\n    pip install -r requirements.txt"
        )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _hr(char: str = "-") -> str:
    return char * min(shutil.get_terminal_size((80, 20)).columns, 88)


def _fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_ingest(args) -> int:
    preflight()
    import dataclasses

    from fkl.config import CONFIG as _BASE
    from fkl.db import connect, stats
    from fkl.pipeline import available_datasets, discover_dataset, ingest_paths

    # Table detection is off by default because it costs ~500x plain text
    # extraction on dense financial reports (see DECISIONS.md).
    CONFIG = dataclasses.replace(_BASE, enable_table_extraction=True) if getattr(
        args, "tables", False
    ) else _BASE
    CONFIG.ensure_dirs()
    if getattr(args, "tables", False):
        print("[ingest] table detection ENABLED - this is substantially slower.")

    paths: list[Path] = [Path(p) for p in args.paths]
    if args.dataset:
        found = discover_dataset(CONFIG, args.dataset)
        if not found:
            names = available_datasets(CONFIG) or ["<none found>"]
            print(
                f"No PDFs matched dataset '{args.dataset}'.\n"
                f"Available datasets in {CONFIG.datasets_dir}: {', '.join(names)}",
                file=sys.stderr,
            )
            return 1
        paths.extend(found)

    if not paths:
        print("Nothing to ingest. Pass PDF paths or --dataset NAME.", file=sys.stderr)
        return 1

    print(f"Ingesting {len(paths)} PDF(s) into {CONFIG.db_path}")
    print(_hr())

    conn = connect(CONFIG.db_path)
    try:
        results = ingest_paths(conn, paths, CONFIG, force=args.force)
    finally:
        conn.close()

    total_time = 0.0
    for r in results:
        total_time += r.elapsed_s
        if r.status == "error":
            print(f"  ERROR  {r.filename}: {r.message}")
            continue
        if r.status == "skipped_unchanged":
            print(f"  SKIP   {r.filename}  (unchanged, {r.chunk_count:,} chunks already stored)")
            continue
        flags = []
        if r.table_chunks:
            flags.append(f"{r.table_chunks} table chunks")
        if r.probably_scanned:
            flags.append("WARNING: little text - possibly scanned/image-only")
        print(
            f"  OK     {r.filename}\n"
            f"         {r.page_count} pages | "
            f"{r.extraction_chunks:,} extraction + {r.retrieval_chunks:,} retrieval chunks | "
            f"page labels resolved: {r.label_coverage:.0%} | {r.elapsed_s:.1f}s"
            + (f"\n         {' | '.join(flags)}" if flags else "")
        )

    print(_hr())
    conn = connect(CONFIG.db_path)
    try:
        s = stats(conn)
    finally:
        conn.close()
    print(
        f"Store now holds {s['documents']} documents, {_fmt_int(s['pages'])} pages, "
        f"{_fmt_int(s['chunks'])} chunks.  ({total_time:.1f}s this run)"
    )
    if s["facts"] == 0:
        print("\nNext: fact extraction (stage 2) turns these chunks into grounded facts.")
    return 0


def cmd_status(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect, json_load, stats

    if not CONFIG.db_path.exists():
        print(f"No database yet at {CONFIG.db_path}. Run:  python run.py ingest --dataset all")
        return 0

    conn = connect(CONFIG.db_path)
    try:
        s = stats(conn)
        docs = conn.execute(
            """SELECT d.*,
                      (SELECT COUNT(*) FROM chunks c
                        WHERE c.document_id = d.id AND c.role='extraction') AS n_extract,
                      (SELECT COUNT(*) FROM chunks c
                        WHERE c.document_id = d.id AND c.role='retrieval')  AS n_retrieve,
                      (SELECT COUNT(*) FROM facts f WHERE f.document_id = d.id) AS n_facts
                 FROM documents d ORDER BY d.ingested_at"""
        ).fetchall()

        print(_hr("="))
        print("FACT KNOWLEDGE LAYER - store status")
        print(_hr("="))
        print(f"database: {CONFIG.db_path}")
        print(
            "totals:   "
            + "  ".join(f"{k}={_fmt_int(v)}" for k, v in s.items())
        )
        print(_hr())
        if not docs:
            print("(no documents ingested yet)")
        for d in docs:
            meta = json_load(d["meta_json"], {}) or {}
            print(f"{d['id']}  {d['filename']}")
            print(
                f"    title: {(d['title'] or '-')[:70]}\n"
                f"    {d['page_count']} pages | {d['n_extract']:,} extraction / "
                f"{d['n_retrieve']:,} retrieval chunks | {d['n_facts']:,} facts | "
                f"status={d['status']}\n"
                f"    printed page labels resolved: "
                f"{meta.get('printed_label_coverage', 0):.0%} | "
                f"tables detected: {meta.get('detected_tables', 0)} | "
                f"parser v{d['parser_version']}"
            )
    finally:
        conn.close()
    return 0


def cmd_doc(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect, json_load

    conn = connect(CONFIG.db_path)
    try:
        d = conn.execute(
            "SELECT * FROM documents WHERE id = ? OR filename LIKE ?",
            (args.document_id, f"%{args.document_id}%"),
        ).fetchone()
        if not d:
            print(f"No document matching '{args.document_id}'", file=sys.stderr)
            return 1
        print(_hr("="))
        for key in ("id", "filename", "title", "page_count", "byte_size", "status",
                    "parser_version", "ingested_at", "content_sha256"):
            print(f"{key:>16}: {d[key]}")
        print(f"{'meta':>16}: {json_load(d['meta_json'], {})}")
        print(_hr())
        pages = conn.execute(
            """SELECT pdf_page_index, printed_page_label, char_count, table_count
                 FROM pages WHERE document_id = ? ORDER BY pdf_page_index LIMIT ?""",
            (d["id"], args.limit),
        ).fetchall()
        print(f"{'pdf_page':>8} {'printed':>8} {'chars':>7} {'tables':>7}")
        for p in pages:
            print(
                f"{p['pdf_page_index']:>8} {str(p['printed_page_label'] or '-'):>8} "
                f"{p['char_count']:>7,} {p['table_count']:>7}"
            )
        if d["page_count"] > args.limit:
            print(f"... ({d['page_count'] - args.limit} more pages; use --limit)")
    finally:
        conn.close()
    return 0


def cmd_page(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect

    conn = connect(CONFIG.db_path)
    try:
        row = conn.execute(
            """SELECT p.* FROM pages p JOIN documents d ON d.id = p.document_id
                WHERE (d.id = ? OR d.filename LIKE ?) AND p.pdf_page_index = ?""",
            (args.document_id, f"%{args.document_id}%", args.page),
        ).fetchone()
        if not row:
            print("No such page.", file=sys.stderr)
            return 1
        print(_hr("="))
        print(
            f"pdf page {row['pdf_page_index']} | printed label: "
            f"{row['printed_page_label'] or '(none detected)'} | "
            f"{row['char_count']:,} chars | {row['table_count']} tables"
        )
        print(_hr())
        print(row["text"][: args.chars])
        if row["char_count"] > args.chars:
            print(f"\n... ({row['char_count'] - args.chars:,} more chars)")
    finally:
        conn.close()
    return 0


def cmd_chunks(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect

    conn = connect(CONFIG.db_path)
    try:
        rows = conn.execute(
            """SELECT c.* FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE (d.id = ? OR d.filename LIKE ?) AND c.role = ?
                ORDER BY c.ordinal LIMIT ?""",
            (args.document_id, f"%{args.document_id}%", args.role, args.limit),
        ).fetchall()
        if not rows:
            print("No chunks matched.", file=sys.stderr)
            return 1
        for c in rows:
            print(_hr("="))
            print(
                f"{c['id']}  [{c['kind']}]  pdf_page={c['pdf_page_index']} "
                f"printed={c['printed_page_label'] or '-'}  "
                f"offsets=[{c['char_start']}:{c['char_end']}] "
                f"verified={bool(c['span_verified'])}  "
                f"~{c['token_estimate']} tok  numeric_density={c['numeric_density']}"
            )
            print(_hr())
            print(c["text"][: args.chars])
    finally:
        conn.close()
    return 0


def cmd_reset(args) -> int:
    from fkl.config import CONFIG

    targets = [CONFIG.db_path, CONFIG.chroma_dir]
    if args.all:
        targets.append(CONFIG.models_dir)
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            print(f"removed directory {t}")
        elif t.exists():
            t.unlink()
            print(f"removed {t}")
        for suffix in ("-wal", "-shm"):
            side = t.with_name(t.name + suffix)
            if side.exists():
                side.unlink()
    print("done.")
    return 0


def cmd_default(args) -> int:
    """`python run.py` with no arguments."""
    preflight()
    from fkl.config import CONFIG
    from fkl.pipeline import available_datasets

    datasets = available_datasets(CONFIG)
    print(_hr("="))
    print("FACT KNOWLEDGE LAYER")
    print(_hr("="))
    print(f"data directory : {CONFIG.data_dir}")
    print(f"datasets found : {', '.join(datasets) if datasets else '(none)'}")
    print(f"llm backend    : {CONFIG.llm_backend}  (stage 2)")
    print(_hr())
    if not datasets:
        print("No datasets found. Ingest PDFs directly:  python run.py ingest FILE.pdf")
        return 0

    target = args.dataset or datasets[0]
    print(f"Ingesting dataset '{target}'.  (use --dataset all for every dataset)\n")
    ingest_args = argparse.Namespace(paths=[], dataset=target, force=False, tables=False)
    return cmd_ingest(ingest_args)


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Fact Knowledge Layer - extract, ground and link facts from PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", help="dataset folder name, a substring of one, or 'all'")
    sub = p.add_subparsers(dest="command")

    ing = sub.add_parser("ingest", help="parse and store PDFs")
    ing.add_argument("paths", nargs="*", help="PDF paths")
    ing.add_argument("--dataset", help="dataset folder name, a substring of one, or 'all'")
    ing.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    ing.add_argument(
        "--tables",
        action="store_true",
        help="also run PyMuPDF table detection (much slower; see DECISIONS.md)",
    )
    ing.set_defaults(func=cmd_ingest)

    st = sub.add_parser("status", help="summarise what is in the store")
    st.set_defaults(func=cmd_status)

    dc = sub.add_parser("doc", help="inspect one document's pages")
    dc.add_argument("document_id")
    dc.add_argument("--limit", type=int, default=25)
    dc.set_defaults(func=cmd_doc)

    pg = sub.add_parser("page", help="print the verbatim text of one page")
    pg.add_argument("document_id")
    pg.add_argument("page", type=int, help="0-based pdf page index")
    pg.add_argument("--chars", type=int, default=3000)
    pg.set_defaults(func=cmd_page)

    ch = sub.add_parser("chunks", help="inspect stored chunks")
    ch.add_argument("document_id")
    ch.add_argument("--role", default="extraction", choices=["extraction", "retrieval"])
    ch.add_argument("--limit", type=int, default=3)
    ch.add_argument("--chars", type=int, default=1200)
    ch.set_defaults(func=cmd_chunks)

    rs = sub.add_parser("reset", help="delete the database (and optionally models)")
    rs.add_argument("--all", action="store_true", help="also delete downloaded models")
    rs.set_defaults(func=cmd_reset)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        return cmd_default(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
