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


class LocalLlamaClient(LLMClient):
    name = "local-llm"
    supports_json_mode = True

    def __init__(self, cfg: Config, *, model_path: Path | None = None, quiet: bool = True):
        from llama_cpp import Llama, LlamaGrammar

        if model_path is None:
            _, filename, model_path = resolve_model(cfg, quiet=quiet)
        else:
            filename = Path(model_path).name

        self.model_name = filename
        self._cfg = cfg
        n_threads = cfg.n_threads or max(1, (os.cpu_count() or 4))

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=cfg.n_ctx,
            n_threads=n_threads,
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
