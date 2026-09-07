"""Similarity search over facts, to surface cross-document candidate pairs.

Why a numpy index rather than Chroma
------------------------------------
The brief allowed either. At this scale the choice is easy: a few thousand facts
times 384 dimensions is a ~5 MB matrix, and a brute-force dot product over it
takes single-digit milliseconds - genuinely faster than an ANN index, and exact
rather than approximate. Vectors are stored in the existing SQLite file, so
there is no second store to keep in sync, no separate persistence directory, and
deleting a document still cascades its vectors away automatically.

The candidate generator only ever pairs facts from *different* documents.
Same-document pairs are not interesting here (a document rarely contradicts
itself in a way the brief cares about) and they would otherwise dominate the
results, since a document repeats its own figures constantly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .embedder import embed_texts


@dataclass
class Candidate:
    fact_a_id: str
    fact_b_id: str
    similarity: float


def _pack(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def index_facts(
    conn: sqlite3.Connection, cfg: Config, *, batch_size: int = 256, on_progress=None
) -> int:
    """Embed every fact that does not yet have a vector. Returns how many."""
    rows = conn.execute(
        """SELECT f.id, f.canonical_text
             FROM facts f
             LEFT JOIN fact_embeddings fe ON fe.fact_id = f.id
            WHERE fe.fact_id IS NULL AND f.canonical_text IS NOT NULL
            ORDER BY f.id"""
    ).fetchall()
    if not rows:
        return 0

    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embed_texts([r["canonical_text"] for r in batch], cfg)
        conn.executemany(
            """INSERT INTO fact_embeddings (fact_id, dim, model, vector)
               VALUES (?,?,?,?)
               ON CONFLICT(fact_id) DO UPDATE SET
                 dim=excluded.dim, model=excluded.model, vector=excluded.vector""",
            [
                (row["id"], int(vec.shape[0]), cfg.embedding_model, _pack(vec))
                for row, vec in zip(batch, vectors)
            ],
        )
        conn.execute(
            "UPDATE facts SET embedded = 1 WHERE id IN ({})".format(
                ",".join("?" * len(batch))
            ),
            [r["id"] for r in batch],
        )
        conn.commit()
        done += len(batch)
        if on_progress:
            on_progress(done, len(rows))
    return done


class VectorIndex:
    """All fact vectors loaded into one matrix, with their document ids."""

    def __init__(self, fact_ids: list[str], document_ids: list[str], matrix: np.ndarray):
        self.fact_ids = fact_ids
        self.document_ids = np.asarray(document_ids)
        self.matrix = matrix
        self._position = {fid: i for i, fid in enumerate(fact_ids)}

    def __len__(self) -> int:
        return len(self.fact_ids)

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "VectorIndex":
        rows = conn.execute(
            """SELECT fe.fact_id, fe.dim, fe.vector, f.document_id
                 FROM fact_embeddings fe
                 JOIN facts f ON f.id = fe.fact_id
                ORDER BY fe.fact_id"""
        ).fetchall()
        if not rows:
            return cls([], [], np.zeros((0, 0), dtype=np.float32))

        dim = rows[0]["dim"]
        matrix = np.empty((len(rows), dim), dtype=np.float32)
        fact_ids, document_ids = [], []
        for i, row in enumerate(rows):
            matrix[i] = _unpack(row["vector"], dim)
            fact_ids.append(row["fact_id"])
            document_ids.append(row["document_id"])
        return cls(fact_ids, document_ids, matrix)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 6,
        threshold: float = 0.0,
        exclude_document_id: str | None = None,
        exclude_fact_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Nearest facts by cosine similarity (vectors are pre-normalised)."""
        if len(self) == 0:
            return []
        scores = self.matrix @ np.asarray(query, dtype=np.float32)

        mask = np.ones(len(self), dtype=bool)
        if exclude_document_id is not None:
            mask &= self.document_ids != exclude_document_id
        if exclude_fact_id is not None and exclude_fact_id in self._position:
            mask[self._position[exclude_fact_id]] = False
        mask &= scores >= threshold

        eligible = np.flatnonzero(mask)
        if eligible.size == 0:
            return []
        order = eligible[np.argsort(-scores[eligible])[:top_k]]
        return [(self.fact_ids[i], float(scores[i])) for i in order]

    def cross_document_pairs(
        self, *, top_k: int = 6, threshold: float = 0.82
    ) -> list[Candidate]:
        """Every above-threshold pair of facts from different documents.

        Each unordered pair is emitted once. Ordering by fact id keeps the run
        deterministic, which matters because the relationship table has a unique
        constraint on (fact_a_id, fact_b_id).
        """
        if len(self) == 0:
            return []

        scores = self.matrix @ self.matrix.T
        # Never pair a fact with itself or with anything in its own document.
        same_doc = self.document_ids[:, None] == self.document_ids[None, :]
        np.fill_diagonal(same_doc, True)
        scores[same_doc] = -1.0

        seen: set[tuple[str, str]] = set()
        candidates: list[Candidate] = []
        for i in range(len(self)):
            row = scores[i]
            eligible = np.flatnonzero(row >= threshold)
            if eligible.size == 0:
                continue
            for j in eligible[np.argsort(-row[eligible])[:top_k]]:
                a, b = sorted((self.fact_ids[i], self.fact_ids[j]))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                candidates.append(Candidate(a, b, float(row[j])))

        candidates.sort(key=lambda c: -c.similarity)
        return candidates
