"""Backend selection.

Resolution order, and the reasoning behind it:

* `LLM_BACKEND=gemini` / `LLM_BACKEND=groq` - explicit opt-in to a cloud
  backend. Errors loudly if no key is present, rather than silently falling
  back, because a user who asked for it wants to know it did not happen.
* `LLM_BACKEND=none` - explicit opt-out of any model. Deterministic extraction.
* `LLM_BACKEND=local` (default) - llama.cpp with a downloaded GGUF. If the model
  cannot be loaded at all, degrade to deterministic extraction with a clear
  warning instead of aborting the run, so a broken download never leaves the
  reviewer with nothing.
"""

from __future__ import annotations

import sys

from ..config import Config
from .base import LLMClient


class BackendUnavailable(RuntimeError):
    pass


def describe_backend(cfg: Config) -> str:
    """One-line summary of what will run, for the CLI banner."""
    backend = (cfg.llm_backend or "local").lower()
    if backend == "gemini":
        from .gemini_client import gemini_available

        return (
            f"gemini ({cfg.gemini_model})"
            if gemini_available()
            else "gemini (NO API KEY - will fail)"
        )
    if backend == "groq":
        from .groq_client import groq_available

        return f"groq ({cfg.groq_model})" if groq_available() else "groq (NO API KEY - will fail)"
    if backend == "none":
        return "none (deterministic pattern extraction, no model)"
    return f"local ({cfg.local_model_file})"


def build_client(cfg: Config, *, quiet: bool = True) -> LLMClient | None:
    """Return a client, or None to signal deterministic (non-LLM) extraction."""
    backend = (cfg.llm_backend or "local").lower()

    if backend == "none":
        return None

    if backend == "gemini":
        from .gemini_client import GeminiClient

        return GeminiClient(cfg)  # raises if the key is missing - intentional

    if backend == "groq":
        from .groq_client import GroqClient

        return GroqClient(cfg)  # raises if the key is missing - intentional

    if backend != "local":
        raise BackendUnavailable(
            f"Unknown LLM_BACKEND={cfg.llm_backend!r}. Use one of: local, gemini, groq, none."
        )

    # Default local path. A GROQ_API_KEY does NOT silently hijack the default;
    # switching backends must be deliberate.
    try:
        from .local_llama import LocalLlamaClient

        return LocalLlamaClient(cfg, quiet=quiet)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not abort
        print(
            f"[llm] could not start the local model ({exc.__class__.__name__}: {exc}).\n"
            "[llm] falling back to deterministic pattern extraction; facts will be "
            "lower quality and are stored with extractor='deterministic'.\n"
            "[llm] to use the fast cloud path instead: export GROQ_API_KEY=... "
            "and set LLM_BACKEND=groq",
            file=sys.stderr,
        )
        return None
