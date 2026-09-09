"""A client that chains several backends, using each until it stops working.

Motivation: the free cloud tiers have hard daily quotas. Once Groq's key is
exhausted (HTTP 429, or an auth/permission error), every subsequent call would
fail the same way, so there is no point retrying it for the rest of the run.
This wrapper catches that failure once, logs it, permanently advances to the
next backend in the chain, and re-issues the same request there.

Configured through `LLM_BACKEND` as a `+`-separated list, e.g.
`LLM_BACKEND=groq+gemini` - "spend the Groq quota first, then fall through to
Gemini". Any ordering of the known cloud backends works.
"""

from __future__ import annotations

import sys

from .base import LLMClient, LLMResponse


def _is_exhaustion(exc: Exception) -> bool:
    """True when the error means 'this key/quota is spent, stop trying it'.

    Covers rate-limit / quota (HTTP 429) and auth/permission failures (401/403)
    across both the Groq SDK (which exposes ``status_code``) and Gemini's raw
    httpx response (``response.status_code``). Any other error - a transient 5xx,
    a network blip - is left to propagate so it is not silently masked.
    """
    for attr in ("status_code", "code"):
        status = getattr(exc, attr, None)
        if status in (401, 403, 429):
            return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in (401, 403, 429):
        return True
    name = exc.__class__.__name__.lower()
    return any(k in name for k in ("ratelimit", "resourceexhausted", "permissiondenied", "authentication"))


class FallbackClient(LLMClient):
    name = "fallback"

    def __init__(self, clients: list[LLMClient], *, quiet: bool = True):
        if not clients:
            raise ValueError("FallbackClient needs at least one backend")
        self._clients = clients
        self._i = 0
        # A backend switch is a rare, once-per-run event that changes which model
        # produced the rest of the facts, so it is always reported, even under
        # `quiet` (which only silences routine per-chunk chatter).
        self._quiet = quiet
        self._sync()

    def _sync(self) -> None:
        active = self._clients[self._i]
        self.model_name = active.model_name
        self.supports_json_mode = active.supports_json_mode

    @property
    def active(self) -> LLMClient:
        return self._clients[self._i]

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> LLMResponse:
        while True:
            try:
                return self.active.complete(
                    system,
                    user,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001 - decide here whether to fall through
                last = self._i >= len(self._clients) - 1
                if last or not _is_exhaustion(exc):
                    raise
                spent = self.active
                self._i += 1
                self._sync()
                print(
                    f"[llm] {spent.name} ({spent.model_name}) is exhausted "
                    f"({exc.__class__.__name__}); switching to {self.active.name} "
                    f"({self.active.model_name}) for the rest of the run.",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
