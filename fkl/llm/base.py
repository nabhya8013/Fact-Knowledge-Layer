"""Backend-agnostic LLM interface.

Every backend - local llama.cpp, Groq, or the deterministic non-LLM fallback -
implements the same tiny surface, so the extraction and linking stages never
know or care which one is active. That is what makes `LLM_BACKEND=local|groq`
a genuine one-line switch rather than a branch scattered through the pipeline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMClient(ABC):
    """A minimal text-in/text-out client.

    `json_mode` asks the backend to guarantee *structurally* valid JSON by
    whatever mechanism it has (a GBNF grammar locally, a response_format flag on
    Groq). It guarantees nothing about the content, so callers still run the
    output through `json_guard`.
    """

    name: str = "base"
    model_name: str = "unknown"
    supports_json_mode: bool = False

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return the model's raw text response."""

    def close(self) -> None:  # pragma: no cover - most backends need nothing
        return None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _Timer:
    """Small helper so every backend reports elapsed time the same way."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0

    @property
    def value(self) -> float:
        return getattr(self, "elapsed", time.perf_counter() - self._t0)
