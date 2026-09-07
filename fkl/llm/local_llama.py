"""Local GGUF backend via llama-cpp-python. The default: no account, no key,
no network after the first run, no separate server process.

Model choice - Qwen2.5-1.5B-Instruct Q4_K_M over Phi-3-mini-4k Q4
----------------------------------------------------------------
* ~1.0 GB versus ~2.3 GB, so the first-run download and the RAM floor are both
  roughly half. On a CPU-only laptop that difference is felt immediately.
* Qwen2.5-1.5B follows "return only JSON, no prose" markedly better; Phi-3 tends
  to prepend an explanation, which then has to be stripped or repaired.
* Under a GBNF grammar the structural gap between the two largely disappears
  anyway, so the remaining differentiators are size and speed.

The grammar is the important part
---------------------------------
Decoding is constrained by a GBNF grammar so the token stream cannot leave the
JSON language. But the grammar describes only the *envelope* - an array of
objects with arbitrary string keys - never a fixed key set. Pinning the keys
would defeat the requirement that the fact schema emerge from the documents, so
structure is enforced and vocabulary is left free.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..config import Config
from .base import LLMClient, LLMResponse

# An array of JSON objects; keys and value shapes are unconstrained.
JSON_ARRAY_GRAMMAR = r"""
root   ::= ws "[" ws (object (ws "," ws object)*)? ws "]" ws
object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair   ::= string ws ":" ws value
value  ::= string | number | object | array | "true" | "false" | "null"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\\x7F\x00-\x1F] | "\\" ["\\bfnrt/] | "\\u" hex hex hex hex
hex    ::= [0-9a-fA-F]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n]*
"""


def _free_ram_gb() -> float:
    """Best-effort free RAM, used to pick between the 1.5B and 0.5B models."""
    try:  # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    try:  # macOS / BSD fallback
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (ValueError, OSError, AttributeError):
        return float("inf")  # unknown: do not downgrade on a guess
    return float("inf")


def resolve_model(cfg: Config, *, quiet: bool = False) -> tuple[str, str, Path]:
    """Choose a model, download it if absent, and return (repo, filename, path).

    Downloads go through `huggingface_hub`, which caches under the project's
    data directory, so the second run is fully offline.
    """
    from huggingface_hub import hf_hub_download

    repo, filename = cfg.local_model_repo, cfg.local_model_file
    free = _free_ram_gb()
    if free < cfg.min_free_ram_gb:
        repo, filename = cfg.fallback_model_repo, cfg.fallback_model_file
        if not quiet:
            print(
                f"[llm] only {free:.1f} GB RAM available (< {cfg.min_free_ram_gb} GB); "
                f"using the smaller {filename}",
                file=sys.stderr,
            )

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.models_dir / filename
    if not target.exists():
        if not quiet:
            print(f"[llm] downloading {filename} from {repo} (first run only)...", file=sys.stderr)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(cfg.models_dir))
    return repo, filename, target


def resolve_threads(cfg: Config) -> int:
    """Thread count, capped.

    Counter-intuitive but measured: this model gets *slower* with more threads
    past ~6 (20 threads ran ~4x slower than 6), because a 1.5B Q4 model is
    memory-bandwidth bound and extra threads just contend. Defaulting to
    `os.cpu_count()` would therefore be a pessimisation on a big machine.
    """
    if cfg.n_threads:
        return cfg.n_threads
    return max(1, min(cfg.n_threads_cap, os.cpu_count() or 4))


def preload_gpu_runtime() -> None:
    """Make pip-installed CUDA runtime libraries loadable without LD_LIBRARY_PATH.

    A CUDA llama-cpp wheel links against libcudart/libcublas, which are not on
    the loader path when they come from the `nvidia-*-cu12` pip packages.
    Setting LD_LIBRARY_PATH after the process has started is too late, so the
    libraries are dlopen'd explicitly with RTLD_GLOBAL before llama_cpp is
    imported; the dynamic linker then resolves libllama's dependencies against
    the already-loaded copies. No-op when the packages are absent, which is the
    normal case for the shipped CPU build.
    """
    import ctypes
    import site

    roots: list[Path] = []
    for base in {*(site.getsitepackages() or []), site.getusersitepackages()}:
        candidate = Path(base) / "nvidia"
        if candidate.is_dir():
            roots.append(candidate)

    # cublas depends on cublasLt, so load in dependency order.
    for pattern in ("cuda_runtime/lib/libcudart.so*", "cublas/lib/libcublasLt.so*",
                    "cublas/lib/libcublas.so*"):
        for root in roots:
            for lib in sorted(root.glob(pattern)):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    continue


def gpu_offload_supported() -> bool:
    """Whether the installed llama-cpp build ships a GPU backend.

    Deliberately a *static* check for the backend shared library rather than a
    call to `llama_supports_gpu_offload()`. That function crashes with SIGILL in
    the prebuilt CUDA wheel used here, and SIGILL cannot be caught in Python -
    it takes the whole process down - so it is not safe to probe at runtime.
    """
    # Locate the package WITHOUT importing it: importing llama_cpp executes the
    # shared-library load, which fails when the CUDA runtime has not been
    # preloaded yet - and this function is what decides whether to preload.
    try:
        import importlib.util

        spec = importlib.util.find_spec("llama_cpp")
        if spec is None or not spec.origin:
            return False
        lib_dir = Path(spec.origin).parent / "lib"
    except Exception:
        return False

    return any(
        any(lib_dir.glob(f"libggml-{backend}.so*")) or any(lib_dir.glob(f"ggml-{backend}.dll"))
        for backend in ("cuda", "metal", "hip", "vulkan", "sycl")
    )


def resolve_gpu_layers(cfg: Config) -> int:
    """Resolve the `-1 => auto` sentinel into a concrete layer count."""
    if cfg.n_gpu_layers >= 0:
        return cfg.n_gpu_layers
    # 999 is llama.cpp's idiom for "offload every layer".
    return 999 if gpu_offload_supported() else 0


class LocalLlamaClient(LLMClient):
    name = "local-llm"
    supports_json_mode = True

    def __init__(self, cfg: Config, *, model_path: Path | None = None, quiet: bool = True):
        preload_gpu_runtime()  # must happen before llama_cpp is imported
        from llama_cpp import Llama, LlamaGrammar

        if model_path is None:
            _, filename, model_path = resolve_model(cfg, quiet=quiet)
        else:
            filename = Path(model_path).name

        self.model_name = filename
        self._cfg = cfg
        self.n_threads = resolve_threads(cfg)
        self.n_gpu_layers = resolve_gpu_layers(cfg)
        self.using_gpu = self.n_gpu_layers != 0

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=cfg.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=512,
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(JSON_ARRAY_GRAMMAR, verbose=False)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            grammar=self._grammar if json_mode else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0
        usage = out.get("usage", {}) or {}
        return LLMResponse(
            text=out["choices"][0]["message"]["content"] or "",
            model=self.model_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            elapsed_s=elapsed,
        )

    def close(self) -> None:
        self._llm = None
