"""Optional Groq backend.

Strictly a bonus path. It is never required, never the default, and is only
reachable when a `GROQ_API_KEY` is present in the environment. With no key the
system runs entirely locally, which is the configuration the grader is expected
to use.

Groq's free tier needs no credit card, and its hosted Llama models are roughly
two orders of magnitude faster than CPU inference, so this is the escape hatch
for anyone who wants the full corpus processed in seconds rather than an hour.
"""

from __future__ import annotations

import os
import time

from ..config import Config
from .base import LLMClient, LLMResponse


def groq_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return key or None


def groq_available() -> bool:
    if not groq_api_key():
        return False
    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return True


class GroqClient(LLMClient):
    name = "groq"
    # Groq's response_format=json_object only accepts a top-level OBJECT and
    # rejects the JSON *array* every prompt here asks for, so we do not use it.
    # json_mode is still honoured as a prompt nudge; json_guard does the parsing
    # and repair, exactly as it does for any non-grammar backend.
    supports_json_mode = True

    def __init__(self, cfg: Config):
        from groq import Groq

        key = groq_api_key()
        if not key:
            raise RuntimeError(
                "LLM_BACKEND=groq but GROQ_API_KEY is not set. "
                "Unset LLM_BACKEND to use the local model instead."
            )
        self._client = Groq(api_key=key)
        self.model_name = cfg.groq_model
        self._cfg = cfg

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if json_mode and "json" not in (system + user).lower():
            user = f"{user}\n\nRespond with a JSON array only."

        t0 = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0
        usage = completion.usage
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model=self.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            elapsed_s=elapsed,
        )
