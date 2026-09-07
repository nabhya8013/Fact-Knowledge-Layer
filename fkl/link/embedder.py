"""Embeddings via fastembed (ONNX), used to find cross-document fact candidates.

Why fastembed rather than sentence-transformers
-----------------------------------------------
Identical models (`bge-small-en-v1.5`, `all-MiniLM-L6-v2`) served through
onnxruntime instead of torch. For a cold clone that must "just work" on CPU,
skipping a 1-2 GB torch download matters more than library familiarity. The
model auto-downloads once and is cached, so later runs are offline.

What gets embedded
------------------
Not the source sentence - the *normalised fact*. Two documents stating the same
thing rarely share wording:

    "revenue from contract with customers was Rs 8,142 crore in FY24"
    "Revenue        81,420      (INR million)"

Embedding the raw text puts those far apart. Embedding
`"Delhivery - revenue: 8142 INR crore [FY24]"` puts them close, which is the
whole point of embedding facts rather than passages.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

import numpy as np

from ..config import Config

_MODEL_CACHE: dict[str, object] = {}


def get_model(cfg: Config):
    """Load (and cache) the embedding model. Downloads once on first use."""
    key = cfg.embedding_model
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from fastembed import TextEmbedding

    cache_dir = cfg.data_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name=key, cache_dir=str(cache_dir))
    _MODEL_CACHE[key] = model
    return model


def embed_texts(texts: Sequence[str], cfg: Config, *, batch_size: int = 64) -> np.ndarray:
    """Embed a batch, returning L2-normalised float32 vectors.

    Normalising here means cosine similarity is a plain dot product later, which
    keeps the search code trivial and avoids repeated norm computation.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = get_model(cfg)
    vectors = np.asarray(list(model.embed(list(texts), batch_size=batch_size)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(text: str, cfg: Config) -> np.ndarray:
    return embed_texts([text], cfg)[0]


def embedding_dim(cfg: Config) -> int:
    return int(embed_one("dimension probe", cfg).shape[0])
