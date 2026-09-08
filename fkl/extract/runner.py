"""Run extraction over many chunks, in parallel where that actually helps.

Why multiprocessing rather than threads
---------------------------------------
A `Llama` instance is not safe to call concurrently, and llama.cpp releases the
GIL during inference anyway, so threads buy nothing. Separate processes each
holding their own model instance do scale - at the cost of one model copy in RAM
per worker, which is why worker count is bounded by free memory as well as by
cores.

Why the GPU path uses a single worker
-------------------------------------
Several processes sharing one GPU serialise on the device and merely add VRAM
pressure and context-switching. Measured, the GPU is already the bottleneck at
one worker, so extra workers make it slower, not faster.

SQLite stays single-writer: workers only run inference and return candidate
records; grounding and all database writes happen in the parent.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import time
from dataclasses import dataclass

from ..config import Config
from ..db import transaction
from ..llm.factory import build_client
from .extractor import (
    ExtractionStats,
    _log_repair,
    mark_chunk_done,
    pending_chunks,
    propose_records,
    store_records,
)

# Rough resident size of one loaded 1.5B Q4 model plus its context, in GB.
_WORKER_RAM_GB = 1.8
_MAX_WORKERS = 4


def _free_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    try:
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (ValueError, OSError, AttributeError):
        return 4.0


def resolve_workers(cfg: Config, requested: int | None = None) -> int:
    """Pick a worker count that will not thrash a small laptop."""
    if requested and requested > 0:
        return requested

    backend = (cfg.llm_backend or "local").lower()
    if backend == "groq":
        # Network-bound, not CPU-bound; a few concurrent requests are fine but
        # the free tier is rate limited, so stay modest.
        return 2
    if backend == "none":
        return 1  # deterministic extraction is already fast

    from ..llm.local_llama import resolve_gpu_layers

    if resolve_gpu_layers(cfg) != 0:
        return 1  # one GPU, already saturated

    cores = os.cpu_count() or 4
    by_cores = max(1, cores // max(1, cfg.n_threads_cap))
    by_ram = max(1, int(_free_ram_gb() // _WORKER_RAM_GB))
    return max(1, min(_MAX_WORKERS, by_cores, by_ram))


# --------------------------------------------------------------------------- #
# Worker process
# --------------------------------------------------------------------------- #

_STATE: dict = {}


def _init_worker(cfg: Config) -> None:
    _STATE["cfg"] = cfg
    _STATE["client"] = build_client(cfg, quiet=True)


def _run_one(job: tuple[str, str, str | None]) -> tuple:
    """Inference only. Returns plain picklable data for the parent to store."""
    chunk_id, chunk_text, title = job
    cfg = _STATE["cfg"]
    client = _STATE["client"]
    try:
        records, guard, extractor_name = propose_records(
            chunk_text, client, cfg, document_title=title
        )
    except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
        return (chunk_id, [], None, "error", 1, f"{exc.__class__.__name__}: {exc}", "")

    if guard is None:
        return (chunk_id, records, "ok_first_try", extractor_name, 1, None, "")
    return (
        chunk_id,
        records if guard.ok else [],
        guard.outcome,
        extractor_name,
        guard.attempts,
        guard.error,
        (guard.raw or "")[:2000] if guard.outcome != "ok_first_try" else "",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class _GuardShim:
    """Carries a worker's JSON outcome into the parent's repair_log writer."""

    outcome: str
    attempts: int
    error: str | None
    raw: str

    @property
    def ok(self) -> bool:
        return self.outcome != "failed"


def run_extraction(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    document_id: str | None = None,
    workers: int | None = None,
    on_progress=None,
    commit_every: int = 10,  # batch SQLite commits: fewer fsyncs; a kill re-does <10 idempotent chunks
    limit: int | None = None,
) -> ExtractionStats:
    """Extract facts for every pending chunk. Safe to re-run; resumes."""
    chunks = pending_chunks(conn, cfg, document_id=document_id)
    if limit:
        chunks = chunks[:limit]
    stats = ExtractionStats()
    if not chunks:
        return stats

    by_id = {c["id"]: c for c in chunks}
    jobs = [(c["id"], c["text"], c["document_title"]) for c in chunks]
    n_workers = resolve_workers(cfg, workers)
    started = time.perf_counter()

    def handle(payload: tuple) -> None:
        chunk_id, records, outcome, extractor_name, attempts, error, raw = payload
        chunk = by_id[chunk_id]
        guard = _GuardShim(outcome or "failed", attempts, error, raw)
        if outcome not in (None, "ok_first_try") or error:
            _log_repair(conn, chunk_id, guard)

        if outcome in ("failed", "error"):
            stats.chunks_seen += 1
            stats.chunks_failed += 1
            stats.repair_outcomes[outcome] += 1
            mark_chunk_done(conn, chunk_id, "failed", 0)
            return

        chunk_stats = store_records(conn, chunk, chunk["page_text"], records, extractor_name)
        chunk_stats.repair_outcomes[outcome or "ok_first_try"] += 1
        stats.merge(chunk_stats)
        mark_chunk_done(conn, chunk_id, "done", chunk_stats.facts_stored)

    if n_workers == 1:
        # No process pool: simpler, and avoids paying model-load cost twice.
        _init_worker(cfg)
        try:
            for index, job in enumerate(jobs, 1):
                handle(_run_one(job))
                if index % commit_every == 0:
                    conn.commit()
                if on_progress:
                    on_progress(index, len(jobs), stats)
        finally:
            client = _STATE.get("client")
            if client:
                client.close()
    else:
        # spawn, not fork: forking a process that has already initialised a GPU
        # context or a loaded model is unreliable.
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for index, payload in enumerate(
                pool.imap_unordered(_run_one, jobs, chunksize=1), 1
            ):
                handle(payload)
                if index % commit_every == 0:
                    conn.commit()
                if on_progress:
                    on_progress(index, len(jobs), stats)

    conn.commit()
    stats.elapsed_s = time.perf_counter() - started

    with transaction(conn):
        if document_id:
            conn.execute(
                "UPDATE documents SET status='extracted' WHERE id=?", (document_id,)
            )
        else:
            conn.execute("UPDATE documents SET status='extracted' WHERE status='parsed'")
    return stats


# Measured on this project's corpus with Qwen2.5-1.5B-Q4_K_M. Used only for the
# up-front estimate, never for control flow.
SECONDS_PER_CHUNK = {
    "none": 0.02,
    "groq": 1.5,
    "local-gpu": 10.4,
    "local-cpu": 22.5,
}


def default_seconds_per_chunk(cfg: Config) -> float:
    backend = (cfg.llm_backend or "local").lower()
    if backend in ("none", "groq"):
        return SECONDS_PER_CHUNK[backend]
    from ..llm.local_llama import resolve_gpu_layers

    return SECONDS_PER_CHUNK["local-gpu" if resolve_gpu_layers(cfg) else "local-cpu"]


def estimate_runtime(conn: sqlite3.Connection, cfg: Config,
                     seconds_per_chunk: float | None = None,
                     *, document_id: str | None = None, workers: int | None = None,
                     limit: int | None = None) -> tuple[int, float, float]:
    """(pending chunks, seconds/chunk used, estimated wall-clock seconds)."""
    pending = len(pending_chunks(conn, cfg, document_id=document_id))
    if limit:
        pending = min(pending, limit)
    per = seconds_per_chunk if seconds_per_chunk else default_seconds_per_chunk(cfg)
    n_workers = resolve_workers(cfg, workers)
    return pending, per, pending * per / max(1, n_workers)
