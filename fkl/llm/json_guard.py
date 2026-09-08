"""Get usable JSON out of a small model, and record honestly how hard it was.

Small quantised models are unreliable JSON emitters. Three defences, applied in
order, with the outcome of every call logged to `repair_log` so the README can
report the real repair rate rather than a guess:

1. **Structural constraint at generation time.** The local backend decodes under
   a GBNF grammar, so the token stream *cannot* leave the JSON language. This is
   far stronger than asking politely in the prompt.
2. **A corrective re-prompt.** If parsing still fails, the model is shown its own
   output and the parser error, and asked to emit only valid JSON.
3. **`json-repair`.** A last-resort mechanical fix for truncation, trailing
   commas, unescaped quotes and similar damage.

Only if all three fail is the chunk abandoned - and that abandonment is counted,
not swallowed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .base import LLMClient

# Models love to wrap JSON in markdown fences despite instructions.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Common keys a model uses when it wraps the array in an object anyway.
_WRAPPER_KEYS = ("facts", "results", "items", "data", "output", "extractions")

OUTCOME_FIRST_TRY = "ok_first_try"
OUTCOME_AFTER_REPROMPT = "ok_after_reprompt"
OUTCOME_AFTER_REPAIR = "ok_after_repair"
OUTCOME_FAILED = "failed"


@dataclass
class GuardResult:
    data: list[dict] | None
    outcome: str
    attempts: int
    error: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.data is not None

    @property
    def needed_repair(self) -> bool:
        return self.outcome in (OUTCOME_AFTER_REPROMPT, OUTCOME_AFTER_REPAIR)


def _strip_fences(text: str) -> str:
    if m := _FENCE_RE.search(text):
        return m.group(1)
    return text.strip()


def _coerce_to_records(value: Any) -> list[dict] | None:
    """Accept the several shapes a model might return for "a list of objects".

    Being liberal here is not sloppiness: the *content* is validated downstream
    (every fact must ground to a real quote), so accepting `{"facts": [...]}`
    instead of `[...]` costs nothing and recovers otherwise-usable extractions.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        for key in _WRAPPER_KEYS:
            inner = value.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        # A single fact returned bare rather than in a list.
        return [value] if value else []
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    return None


def _try_parse(text: str) -> tuple[list[dict] | None, str | None]:
    cleaned = _strip_fences(text)
    if not cleaned.strip():
        return [], None  # an empty response legitimately means "no facts here"
    try:
        return _coerce_to_records(json.loads(cleaned)), None
    except (ValueError, TypeError) as exc:
        return None, str(exc)


def _try_repair(text: str) -> tuple[list[dict] | None, str | None]:
    try:
        from json_repair import repair_json
    except ImportError:  # pragma: no cover - dependency is pinned
        return None, "json-repair not installed"
    try:
        repaired = repair_json(_strip_fences(text), return_objects=True)
    except Exception as exc:
        return None, f"json-repair failed: {exc}"
    records = _coerce_to_records(repaired)
    if records is None:
        return None, "json-repair produced a non-record shape"
    return records, None


REPROMPT_TEMPLATE = (
    "Your previous reply could not be parsed as JSON.\n"
    "Parser error: {error}\n\n"
    "Your previous reply was:\n{previous}\n\n"
    "Reply again with ONLY the corrected JSON array. "
    "No explanation, no markdown fences, no trailing text."
)


def guarded_json(
    client: LLMClient,
    system: str,
    user: str,
    *,
    max_retries: int = 1,
    max_tokens: int = 768,
    temperature: float = 0.0,
    log: Callable[[GuardResult], None] | None = None,
) -> GuardResult:
    """Call the model and return parsed records, repairing if necessary."""
    attempts = 0
    response = client.complete(
        system,
        user,
        json_mode=client.supports_json_mode,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    attempts += 1
    raw = response.text

    data, error = _try_parse(raw)
    if data is not None:
        result = GuardResult(data, OUTCOME_FIRST_TRY, attempts, None, raw)
        if log:
            log(result)
        return result

    # Tier 2: mechanical repair BEFORE spending another inference call. Under the
    # GBNF grammar the first reply is always syntactically valid JSON, so a parse
    # failure here means truncation at max_tokens - json-repair closes the open
    # array and drops the single half-written trailing object, keeping every
    # fact already emitted. A corrective re-prompt would cost a second full
    # generation to recover the same records.
    data, repair_error = _try_repair(raw)
    if data is not None:
        result = GuardResult(data, OUTCOME_AFTER_REPAIR, attempts, error, raw)
        if log:
            log(result)
        return result

    # Tier 3: show the model its own broken output and ask again.
    for _ in range(max_retries):
        retry = client.complete(
            system,
            REPROMPT_TEMPLATE.format(error=error, previous=raw[:2000]),
            json_mode=client.supports_json_mode,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        attempts += 1
        raw = retry.text
        data, error = _try_parse(raw)
        if data is not None:
            result = GuardResult(data, OUTCOME_AFTER_REPROMPT, attempts, None, raw)
            if log:
                log(result)
            return result

    # Last chance: repair whatever the final re-prompt returned.
    data, repair_error = _try_repair(raw)
    if data is not None:
        result = GuardResult(data, OUTCOME_AFTER_REPAIR, attempts, error, raw)
        if log:
            log(result)
        return result

    result = GuardResult(None, OUTCOME_FAILED, attempts, repair_error or error, raw)
    if log:
        log(result)
    return result
