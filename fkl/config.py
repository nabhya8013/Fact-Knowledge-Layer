"""Central configuration for the Fact Knowledge Layer.

Every tunable lives here and every value is overridable through an environment
variable. Nothing in this file (or anywhere else in the codebase) encodes
knowledge about a *specific* document: the defaults are generic thresholds, not
per-PDF rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _env_path("FKL_DATA_DIR", PROJECT_ROOT / "data"))
    datasets_dir: Path = field(
        default_factory=lambda: _env_path("FKL_DATASETS_DIR", PROJECT_ROOT / "starter-datasets")
    )

    # ------------------------------------------------------------- ingestion
    # An extraction unit is normally a whole page. Pages longer than this get
    # split at paragraph/sentence boundaries so we never blow the model context.
    max_extraction_tokens: int = field(
        default_factory=lambda: _env_int("FKL_MAX_EXTRACTION_TOKENS", 900)
    )
    # Retrieval chunks are a separate, overlapping view of the same text, used
    # only for embedding/similarity - not for extraction.
    retrieval_chunk_chars: int = field(
        default_factory=lambda: _env_int("FKL_RETRIEVAL_CHUNK_CHARS", 900)
    )
    retrieval_chunk_overlap: int = field(
        default_factory=lambda: _env_int("FKL_RETRIEVAL_CHUNK_OVERLAP", 150)
    )
    # Chunks shorter than this are dropped as noise (page furniture, stray
    # headers left behind after a split).
    min_chunk_chars: int = field(default_factory=lambda: _env_int("FKL_MIN_CHUNK_CHARS", 80))

    # Table handling. OFF by default, and that is a measured decision rather than
    # laziness: PyMuPDF's detector costs ~150s on a 100-page financial report
    # versus ~0.3s for plain text extraction (a ~500x penalty), while PyMuPDF's
    # plain page text already linearises those tables into readable row-major
    # order. Enable with FKL_ENABLE_TABLES=1 or `--tables` when table structure
    # matters more than ingest speed. See DECISIONS.md.
    enable_table_extraction: bool = field(
        default_factory=lambda: _env_bool("FKL_ENABLE_TABLES", False)
    )
    table_strategy: str = field(default_factory=lambda: _env_str("FKL_TABLE_STRATEGY", "lines"))
    table_min_rows: int = field(default_factory=lambda: _env_int("FKL_TABLE_MIN_ROWS", 2))
    table_min_cols: int = field(default_factory=lambda: _env_int("FKL_TABLE_MIN_COLS", 2))
    # Cheap pre-check before paying for PyMuPDF's table detector: a page needs at
    # least this many predominantly-numeric lines to be worth probing.
    table_probe_min_numeric_lines: int = field(
        default_factory=lambda: _env_int("FKL_TABLE_PROBE_MIN_LINES", 4)
    )

    # Extraction is expensive on CPU. Units with essentially no numbers and no
    # substantive prose are skipped. This is a generic density heuristic, not a
    # document-specific filter.
    min_numeric_density: float = field(
        default_factory=lambda: _env_float("FKL_MIN_NUMERIC_DENSITY", 0.02)
    )

    # ------------------------------------------------------------------- llm
    # local  -> llama-cpp-python + a quantised GGUF, downloaded on first run
    # groq   -> Groq free API, only when GROQ_API_KEY is present
    # none   -> deterministic (non-LLM) extractor; guarantees the pipeline runs
    llm_backend: str = field(default_factory=lambda: _env_str("LLM_BACKEND", "local"))
    local_model_repo: str = field(
        default_factory=lambda: _env_str("FKL_MODEL_REPO", "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    )
    local_model_file: str = field(
        default_factory=lambda: _env_str("FKL_MODEL_FILE", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    )
    # Automatically used instead of the above when free RAM is tight.
    fallback_model_repo: str = field(
        default_factory=lambda: _env_str("FKL_FALLBACK_MODEL_REPO", "bartowski/Qwen2.5-0.5B-Instruct-GGUF")
    )
    fallback_model_file: str = field(
        default_factory=lambda: _env_str("FKL_FALLBACK_MODEL_FILE", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    )
    min_free_ram_gb: float = field(default_factory=lambda: _env_float("FKL_MIN_FREE_RAM_GB", 3.0))

    n_ctx: int = field(default_factory=lambda: _env_int("FKL_N_CTX", 4096))
    # 0 => auto. Measured: throughput plateaus around 6 threads and *degrades*
    # badly beyond that (20 threads was ~4x slower than 6 on this model), so
    # "use every core" is actively wrong here. See DECISIONS.md.
    n_threads: int = field(default_factory=lambda: _env_int("FKL_N_THREADS", 0))
    n_threads_cap: int = field(default_factory=lambda: _env_int("FKL_N_THREADS_CAP", 6))
    # GPU offload. -1 => auto-detect: offload everything when the installed
    # llama-cpp build supports it, otherwise run on CPU. The shipped
    # requirements.txt installs the CPU build, so this resolves to 0 for a
    # normal clone; installing a CUDA/Metal wheel turns it on with no code
    # change. Set FKL_N_GPU_LAYERS=0 to force CPU even on a GPU machine.
    n_gpu_layers: int = field(default_factory=lambda: _env_int("FKL_N_GPU_LAYERS", -1))
    max_output_tokens: int = field(default_factory=lambda: _env_int("FKL_MAX_OUTPUT_TOKENS", 768))
    temperature: float = field(default_factory=lambda: _env_float("FKL_TEMPERATURE", 0.0))

    groq_model: str = field(
        default_factory=lambda: _env_str("FKL_GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # How many corrective re-prompts before falling through to json-repair.
    json_max_retries: int = field(default_factory=lambda: _env_int("FKL_JSON_MAX_RETRIES", 1))

    # --------------------------------------------------------------- linking
    embedding_model: str = field(
        default_factory=lambda: _env_str("FKL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("FKL_SIMILARITY_THRESHOLD", 0.62)
    )
    candidate_top_k: int = field(default_factory=lambda: _env_int("FKL_CANDIDATE_TOP_K", 6))

    # ------------------------------------------------------------------- api
    host: str = field(default_factory=lambda: _env_str("FKL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("FKL_PORT", 8000))

    # ----------------------------------------------------------- derivations
    @property
    def db_path(self) -> Path:
        return self.data_dir / "fkl.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.chroma_dir, self.uploads_dir):
            p.mkdir(parents=True, exist_ok=True)


CONFIG = Config()

# Bumped whenever parsing/chunking changes in a way that invalidates stored
# chunks. Ingestion compares this against the value recorded on the document and
# re-parses automatically when they differ (see pipeline.ingest_pdf).
PARSER_VERSION = "1"
