#!/usr/bin/env python3
"""Fact Knowledge Layer - single entry point.

    python run.py                     # ingest -> extract -> link -> serve the UI
    python run.py ingest --dataset all
    python run.py ingest path/to/a.pdf path/to/b.pdf
    python run.py status
    python run.py doc <document_id>
    python run.py page <document_id> <pdf_page_index>
    python run.py extract [--limit N]  # grounded fact extraction (resumable)
    python run.py link                # cross-document relationships
    python run.py facts | relations | schema
    python run.py serve               # web UI + API only
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


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else width
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def cmd_extract(args) -> int:
    preflight()
    import time

    from fkl.config import CONFIG
    from fkl.db import connect, stats as db_stats
    from fkl.extract.runner import estimate_runtime, resolve_workers, run_extraction
    from fkl.llm.factory import describe_backend

    CONFIG.ensure_dirs()
    conn = connect(CONFIG.db_path)
    try:
        document_id = None
        if args.document_id:
            row = conn.execute(
                "SELECT id FROM documents WHERE id = ? OR filename LIKE ?",
                (args.document_id, f"%{args.document_id}%"),
            ).fetchone()
            if not row:
                print(f"No document matching '{args.document_id}'", file=sys.stderr)
                return 1
            document_id = row["id"]

        if args.redo:
            if document_id:
                chunk_ids = "(SELECT id FROM chunks WHERE document_id = ?)"
                conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
                conn.execute(
                    f"DELETE FROM extraction_progress WHERE chunk_id IN {chunk_ids}",
                    (document_id,),
                )
                conn.execute(
                    f"DELETE FROM repair_log WHERE stage='extraction' AND chunk_id IN {chunk_ids}",
                    (document_id,),
                )
            else:
                conn.execute("DELETE FROM facts")
                conn.execute("DELETE FROM extraction_progress")
                conn.execute("DELETE FROM fact_types")
                # Otherwise the repair rate is measured against a stale denominator.
                conn.execute("DELETE FROM repair_log WHERE stage = 'extraction'")
            conn.commit()
            print("[extract] --redo: cleared existing facts for re-extraction")

        workers = resolve_workers(CONFIG, args.workers)
        pending, per_chunk, est_s = estimate_runtime(
            conn, CONFIG, args.seconds_per_chunk, document_id=document_id,
            workers=args.workers, limit=args.limit,
        )

        print(_hr("="))
        print("FACT EXTRACTION")
        print(_hr("="))
        print(f"backend        : {describe_backend(CONFIG)}")
        print(f"workers        : {workers}")
        print(f"pending chunks : {pending:,}")
        if pending == 0:
            print("\nNothing to do - every chunk has already been extracted.")
            print("Use --redo to discard existing facts and extract again.")
            return 0
        print(
            f"estimated time : ~{est_s/60:.1f} min "
            f"(at {per_chunk:.1f}s/chunk, measured for this backend)"
        )
        print("               progress is saved continuously - Ctrl-C and re-run to resume")
        print(_hr())

        start = time.perf_counter()
        last_line = [0.0]

        def on_progress(done, total, running):
            now = time.perf_counter()
            if now - last_line[0] < 1.0 and done != total:
                return
            last_line[0] = now
            elapsed = now - start
            rate = done / elapsed if elapsed else 0
            remaining = (total - done) / rate if rate else 0
            sys.stdout.write(
                f"\r  {_progress_bar(done, total)}  "
                f"{running.facts_stored:,} facts  "
                f"{elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m left   "
            )
            sys.stdout.flush()

        try:
            result = run_extraction(
                conn, CONFIG, document_id=document_id, workers=args.workers,
                on_progress=on_progress, limit=args.limit,
            )
        except KeyboardInterrupt:
            conn.commit()
            print("\n\nInterrupted. Progress saved - re-run `python run.py extract` to resume.")
            return 130

        print("\n" + _hr())
        print(
            f"chunks processed : {result.chunks_processed:,} "
            f"({result.chunks_failed:,} failed)\n"
            f"facts proposed   : {result.facts_proposed:,}\n"
            f"facts stored     : {result.facts_stored:,} "
            f"(grounding rate {result.grounding_rate:.0%})\n"
            f"  rejected       : {result.rejected_ungrounded:,} ungrounded, "
            f"{result.rejected_invalid:,} malformed, {result.rejected_duplicate:,} duplicate\n"
            f"elapsed          : {result.elapsed_s/60:.1f} min"
        )
        if result.match_modes:
            print("evidence match modes:")
            for mode, count in result.match_modes.most_common():
                print(f"    {mode:24s} {count:,}")
        if result.repair_outcomes:
            print(f"JSON repair path fired on {result.repair_rate:.1%} of chunks:")
            for outcome, count in result.repair_outcomes.most_common():
                print(f"    {outcome:24s} {count:,}")

        s = db_stats(conn)
        print(_hr())
        print(f"store: {s['facts']:,} facts, {s['evidence']:,} evidence rows, "
              f"{s['fact_types']:,} fact types")
    finally:
        conn.close()
    return 0


def cmd_facts(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect, json_load

    conn = connect(CONFIG.db_path)
    try:
        sql = """SELECT f.*, e.quote, e.pdf_page_index, e.printed_page_label, e.match_mode,
                        d.filename
                   FROM facts f
                   JOIN evidence e ON e.fact_id = f.id
                   JOIN documents d ON d.id = f.document_id
                  WHERE 1=1"""
        params: list = []
        if args.document_id:
            sql += " AND (d.id = ? OR d.filename LIKE ?)"
            params += [args.document_id, f"%{args.document_id}%"]
        if args.type:
            sql += " AND f.fact_type LIKE ?"
            params.append(f"%{args.type}%")
        if args.numeric_only:
            sql += " AND f.value_num IS NOT NULL"
        sql += " ORDER BY f.confidence DESC LIMIT ?"
        params.append(args.limit)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            print("No facts matched. Run `python run.py extract` first?")
            return 0
        for r in rows:
            payload = json_load(r["payload_json"], {}) or {}
            page = r["printed_page_label"] or f"pdf#{r['pdf_page_index']}"
            print(_hr("="))
            print(
                f"{payload.get('subject')} | {payload.get('attribute')} = "
                f"{payload.get('value')} {payload.get('unit') or ''}"
                + (f"  [{payload.get('time_scope')}]" if payload.get("time_scope") else "")
            )
            print(
                f"  type={r['fact_type']}  value_num={r['value_num']}  unit={r['unit']}  "
                f"conf={r['confidence']:.2f}  by={r['extractor']}"
            )
            extra = {k: v for k, v in payload.items() if k not in
                     ("subject", "attribute", "value", "unit", "time_scope",
                      "qualifier", "confidence", "source_quote")}
            if extra:
                print(f"  emergent keys: {extra}")
            print(f"  EVIDENCE {r['filename']} p.{page} ({r['match_mode']}):")
            print(f"    \"{(r['quote'] or '')[:220]}\"")
    finally:
        conn.close()
    return 0


def cmd_schema(args) -> int:
    """Show the dynamic fact-type registry - the schema as it actually emerged."""
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect, json_load

    conn = connect(CONFIG.db_path)
    try:
        rows = conn.execute(
            """SELECT ft.*, d.filename FROM fact_types ft
               LEFT JOIN documents d ON d.id = ft.first_seen_document_id
               ORDER BY ft.fact_count DESC LIMIT ?""",
            (args.limit,),
        ).fetchall()
        if not rows:
            print("No fact types yet. Run `python run.py extract` first.")
            return 0
        total = conn.execute("SELECT COUNT(*) FROM fact_types").fetchone()[0]
        print(_hr("="))
        print(f"DYNAMIC FACT SCHEMA - {total:,} types discovered from the documents")
        print(_hr("="))
        for r in rows:
            keys = json_load(r["observed_keys_json"], []) or []
            print(f"{r['fact_count']:>5}x  {r['name']}")
            print(f"         keys: {', '.join(keys)}")
            print(f"         first seen in: {r['filename'] or '-'} at {r['first_seen_at']}")
    finally:
        conn.close()
    return 0


def cmd_link(args) -> int:
    preflight()
    import time

    from fkl.config import CONFIG
    from fkl.db import connect
    from fkl.link.relate import link_facts
    from fkl.link.vector_index import VectorIndex, index_facts
    from fkl.llm.factory import build_client, describe_backend

    CONFIG.ensure_dirs()
    conn = connect(CONFIG.db_path)
    try:
        n_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if n_facts == 0:
            print("No facts yet. Run `python run.py extract` first.")
            return 1

        print(_hr("="))
        print("CROSS-DOCUMENT LINKING")
        print(_hr("="))
        print(f"backend   : {describe_backend(CONFIG)}")
        print(f"embedding : {CONFIG.embedding_model}")
        print(f"threshold : {args.threshold or CONFIG.similarity_threshold}")
        print(_hr())

        print("embedding facts...", end=" ", flush=True)
        t0 = time.perf_counter()
        embedded = index_facts(conn, CONFIG)
        print(f"{embedded:,} newly embedded in {time.perf_counter()-t0:.1f}s")

        index = VectorIndex.load(conn)
        pairs = index.cross_document_pairs(
            top_k=args.top_k or CONFIG.candidate_top_k,
            threshold=args.threshold or CONFIG.similarity_threshold,
        )
        print(f"index holds {len(index):,} vectors -> {len(pairs):,} cross-document candidate pairs")
        if args.dry_run:
            for c in pairs[:20]:
                print(f"  {c.similarity:.3f}  {c.fact_a_id}  <->  {c.fact_b_id}")
            return 0
        if not pairs:
            print("\nNo candidates above the threshold. Try --threshold lower.")
            return 0
        print(_hr())

        client = build_client(CONFIG, quiet=False)
        start = time.perf_counter()
        last = [0.0]

        def on_progress(done, total, running):
            now = time.perf_counter()
            if now - last[0] < 1.0 and done != total:
                return
            last[0] = now
            elapsed = now - start
            rate = done / elapsed if elapsed else 0
            sys.stdout.write(
                f"\r  {_progress_bar(done, total)}  {running.stored:,} stored  "
                f"~{((total-done)/rate/60) if rate else 0:.1f}m left   "
            )
            sys.stdout.flush()

        try:
            stats = link_facts(
                conn, CONFIG, client,
                threshold=args.threshold, top_k=args.top_k, limit=args.limit,
                on_progress=on_progress,
            )
        finally:
            if client:
                client.close()

        print("\n" + _hr())
        print(
            f"candidates : {stats.candidates:,}\n"
            f"classified : {stats.classified:,}  (skipped {stats.skipped_existing:,} already linked)\n"
            f"stored     : {stats.stored:,}\n"
            f"elapsed    : {stats.elapsed_s/60:.1f} min"
        )
        print("\nrelation types:")
        for name, count in stats.by_type.most_common():
            print(f"    {name:22s} {count:,}")
        print("top reasons:")
        for name, count in stats.by_reason.most_common(8):
            print(f"    {name:22s} {count:,}")
    finally:
        conn.close()
    return 0


def cmd_relations(args) -> int:
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect
    from fkl.link.relate import load_fact_view

    conn = connect(CONFIG.db_path)
    try:
        sql = "SELECT * FROM relationships WHERE 1=1"
        params: list = []
        if args.type:
            sql += " AND relation_type = ?"
            params.append(args.type.upper())
        if not args.include_unrelated and not args.type:
            sql += " AND relation_type != 'UNRELATED'"
        sql += " ORDER BY confidence DESC, similarity_score DESC LIMIT ?"
        params.append(args.limit)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            print("No relationships. Run `python run.py link` first.")
            return 0
        for r in rows:
            a = load_fact_view(conn, r["fact_a_id"])
            b = load_fact_view(conn, r["fact_b_id"])
            if not a or not b:
                continue
            print(_hr("="))
            print(
                f"{r['relation_type']}  ({r['reason_tag']})  "
                f"confidence={r['confidence']:.2f}  similarity={r['similarity_score']:.3f}"
            )
            print(f"  -> {r['explanation']}")
            for label, f in (("A", a), ("B", b)):
                print(
                    f"  [{label}] {f['subject']} | {f['attribute']} = "
                    f"{f['value']} {f['unit'] or ''} [{f['time_scope']}]"
                )
                print(f"      {f['filename']} p.{f['page']}")
                print(f"      \"{(f['source_quote'] or '')[:150]}\"")
    finally:
        conn.close()
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

    targets = [CONFIG.db_path, CONFIG.data_dir / "embeddings"]
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


def cmd_serve(args) -> int:
    preflight([("fitz", "pymupdf"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn")])
    import uvicorn

    from fkl.config import CONFIG
    from fkl.db import connect, stats as db_stats

    CONFIG.ensure_dirs()
    conn = connect(CONFIG.db_path)
    try:
        s = db_stats(conn)
    finally:
        conn.close()

    host, port = args.host or CONFIG.host, args.port or CONFIG.port
    print(_hr("="))
    print("FACT KNOWLEDGE LAYER - web UI")
    print(_hr("="))
    print(f"  UI       http://{host}:{port}/")
    print(f"  API docs http://{host}:{port}/docs")
    print(f"  showcase http://{host}:{port}/api/showcase")
    print(f"\n  store: {s['documents']} documents, {s['facts']:,} facts, "
          f"{s['relationships']:,} relationships")
    if s["relationships"] == 0:
        print("  (no relationships yet - run `python run.py link`)")
    print(_hr())
    uvicorn.run("fkl.api:app", host=host, port=port, log_level="warning")
    return 0


def cmd_default(args) -> int:
    """`python run.py` - the one command a reviewer runs.

    Ingest, extract, link, then serve. Each stage is resumable and skips work
    that is already done, so re-running is cheap and interrupting is safe.
    """
    preflight()
    from fkl.config import CONFIG
    from fkl.db import connect, stats as db_stats
    from fkl.pipeline import available_datasets

    CONFIG.ensure_dirs()
    datasets = available_datasets(CONFIG)

    print(_hr("="))
    print("FACT KNOWLEDGE LAYER")
    print(_hr("="))
    print(f"data directory : {CONFIG.data_dir}")
    print(f"datasets found : {', '.join(datasets) if datasets else '(none)'}")
    print(_hr())

    if datasets:
        target = args.dataset or "all"
        rc = cmd_ingest(argparse.Namespace(
            paths=[], dataset=target, force=False, tables=False))
        if rc != 0:
            return rc

    if not args.no_extract:
        rc = cmd_extract(argparse.Namespace(
            document_id=None, workers=None, seconds_per_chunk=None,
            redo=False, limit=args.limit))
        if rc not in (0, 130):
            return rc
        if rc == 130:
            return rc

        conn = connect(CONFIG.db_path)
        try:
            has_facts = db_stats(conn)["facts"] > 0
        finally:
            conn.close()
        if has_facts:
            rc = cmd_link(argparse.Namespace(
                threshold=None, top_k=None, limit=None, dry_run=False))
            if rc != 0:
                return rc

    return cmd_serve(argparse.Namespace(host=None, port=None))


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Fact Knowledge Layer - extract, ground and link facts from PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", help="dataset folder name, a substring of one, or 'all'")
    p.add_argument("--no-extract", action="store_true",
                   help="ingest and serve without running extraction")
    p.add_argument("--limit", type=int, default=None,
                   help="cap extraction chunks (quick demo run)")
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

    ex = sub.add_parser("extract", help="extract grounded facts from ingested chunks")
    ex.add_argument("document_id", nargs="?", help="limit to one document")
    ex.add_argument("--workers", type=int, default=None, help="override worker count")
    ex.add_argument("--seconds-per-chunk", type=float, default=None,
                    help="override the up-front estimate only (not control flow)")
    ex.add_argument("--limit", type=int, default=None,
                    help="stop after N chunks (for quick iteration)")
    ex.add_argument("--redo", action="store_true", help="discard existing facts and re-extract")
    ex.set_defaults(func=cmd_extract)

    fa = sub.add_parser("facts", help="browse extracted facts with their evidence")
    fa.add_argument("document_id", nargs="?")
    fa.add_argument("--type", help="filter by fact type substring")
    fa.add_argument("--numeric-only", action="store_true")
    fa.add_argument("--limit", type=int, default=15)
    fa.set_defaults(func=cmd_facts)

    lk = sub.add_parser("link", help="embed facts and classify cross-document relationships")
    lk.add_argument("--threshold", type=float, default=None, help="cosine similarity cutoff")
    lk.add_argument("--top-k", type=int, default=None, help="neighbours considered per fact")
    lk.add_argument("--limit", type=int, default=None, help="stop after N pairs")
    lk.add_argument("--dry-run", action="store_true", help="show candidate pairs, do not classify")
    lk.set_defaults(func=cmd_link)

    rl = sub.add_parser("relations", help="browse classified relationships with evidence")
    rl.add_argument("--type", help="CORROBORATES | CONTRADICTS | CONTEXT_RECONCILED | UNRELATED")
    rl.add_argument("--include-unrelated", action="store_true")
    rl.add_argument("--limit", type=int, default=10)
    rl.set_defaults(func=cmd_relations)

    sc = sub.add_parser("schema", help="show the dynamic fact-type registry")
    sc.add_argument("--limit", type=int, default=30)
    sc.set_defaults(func=cmd_schema)

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

    sv = sub.add_parser("serve", help="run the web UI and API")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.set_defaults(func=cmd_serve)

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
