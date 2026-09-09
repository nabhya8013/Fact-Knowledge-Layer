"""Tests for the LLM layer: JSON recovery, backend selection, deterministic path.

These use a scripted fake client rather than a real model. The point is to prove
the *recovery machinery* behaves correctly on each class of malformed output -
which a real small model produces only intermittently, and therefore cannot be
relied on to exercise every branch in a test run.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.config import CONFIG  # noqa: E402
from fkl.llm.base import LLMClient, LLMResponse  # noqa: E402
from fkl.llm.deterministic import deterministic_facts, deterministic_relation  # noqa: E402
from fkl.llm.json_guard import (  # noqa: E402
    OUTCOME_AFTER_REPAIR,
    OUTCOME_AFTER_REPROMPT,
    OUTCOME_FAILED,
    OUTCOME_FIRST_TRY,
    _coerce_to_records,
    guarded_json,
)


class ScriptedClient(LLMClient):
    """Returns pre-scripted responses in order, recording the prompts it saw."""

    name = "scripted"
    model_name = "scripted"
    supports_json_mode = True

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, json_mode=False, max_tokens=768, temperature=0.0):
        self.calls.append((system, user))
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, model=self.model_name)


# --------------------------------------------------------------------------- #
# JSON recovery tiers
# --------------------------------------------------------------------------- #


def test_clean_json_parses_on_the_first_try():
    client = ScriptedClient(['[{"subject":"A","attribute":"x","value":"1"}]'])
    result = guarded_json(client, "sys", "user")
    assert result.ok and result.outcome == OUTCOME_FIRST_TRY
    assert result.attempts == 1 and len(client.calls) == 1
    assert result.data[0]["subject"] == "A"


def test_markdown_fences_are_stripped_without_a_retry():
    client = ScriptedClient(['```json\n[{"subject":"A"}]\n```'])
    result = guarded_json(client, "sys", "user")
    assert result.ok and result.outcome == OUTCOME_FIRST_TRY
    assert len(client.calls) == 1, "fence stripping must not cost a model call"


def test_broken_json_triggers_a_corrective_reprompt():
    client = ScriptedClient(["not json at all", '[{"subject":"B"}]'])
    result = guarded_json(client, "sys", "user", max_retries=1)
    assert result.ok and result.outcome == OUTCOME_AFTER_REPROMPT
    assert result.attempts == 2 and len(client.calls) == 2
    # The retry must show the model its own output and the parser error.
    retry_prompt = client.calls[1][1]
    assert "not json at all" in retry_prompt
    assert "could not be parsed" in retry_prompt


def test_truncated_json_falls_through_to_repair():
    """The dominant real failure mode: output cut off at max_tokens mid-string."""
    truncated = '[{"subject":"RBI","attribute":"inflation","value":"4.6","source_quote":"moderated to 4.6'
    client = ScriptedClient([truncated, truncated])  # reprompt fails too
    result = guarded_json(client, "sys", "user", max_retries=1)
    assert result.ok and result.outcome == OUTCOME_AFTER_REPAIR
    assert result.data[0]["subject"] == "RBI"
    assert result.data[0]["value"] == "4.6"


def test_unrecoverable_output_is_reported_not_swallowed():
    client = ScriptedClient(["<<<garbage", "<<<still garbage"])
    result = guarded_json(client, "sys", "user", max_retries=1)
    assert not result.ok
    assert result.outcome == OUTCOME_FAILED
    assert result.data is None


def test_empty_response_means_no_facts_not_an_error():
    """A chunk with nothing extractable is a normal outcome, not a failure."""
    client = ScriptedClient([""])
    result = guarded_json(client, "sys", "user")
    assert result.ok and result.data == []


def test_repair_outcomes_are_logged_through_the_callback():
    seen = []
    client = ScriptedClient(["bad", '[{"subject":"C"}]'])
    guarded_json(client, "sys", "user", max_retries=1, log=seen.append)
    assert len(seen) == 1 and seen[0].outcome == OUTCOME_AFTER_REPROMPT


@pytest.mark.parametrize(
    "value,expected_len",
    [
        ([{"a": 1}, {"b": 2}], 2),
        ({"facts": [{"a": 1}]}, 1),          # model wrapped the array in an object
        ({"results": [{"a": 1}, {"b": 2}]}, 2),
        ({"subject": "solo"}, 1),            # single fact returned bare
        ([{"a": 1}, "junk", 5], 1),          # non-dict entries dropped
    ],
)
def test_record_coercion_accepts_the_shapes_models_actually_return(value, expected_len):
    records = _coerce_to_records(value)
    assert records is not None and len(records) == expected_len


# --------------------------------------------------------------------------- #
# Deterministic fallback
# --------------------------------------------------------------------------- #


def test_deterministic_extractor_quotes_verbatim():
    """Its quotes must ground, or the fallback would produce nothing usable."""
    from fkl.extract.grounding import find_quote

    text = (
        "Revenue from operations stood at Rs 8,142 crore in FY24. "
        "The company operated 42 distribution centres as of March 31, 2024."
    )
    facts = deterministic_facts(text, document_title="Test Co")
    assert facts, "should find quantitative statements"
    for fact in facts:
        assert find_quote(text, fact["source_quote"]) is not None
        assert fact["confidence"] < 0.5, "pattern matching must not claim high confidence"


def test_deterministic_extractor_ignores_text_without_quantities():
    facts = deterministic_facts("The board met to discuss strategy and governance matters.")
    assert facts == []


def test_deterministic_relation_detects_unit_difference():
    a = {"subject": "X", "attribute": "revenue", "value": "8142", "unit": "INR crore"}
    b = {"subject": "X", "attribute": "revenue", "value": "1200", "unit": "USD million"}
    verdict = deterministic_relation(a, b)
    assert verdict["relation_type"] == "CONTEXT_RECONCILED"
    assert verdict["reason_tag"] == "units_differ"


def test_deterministic_relation_flags_a_true_contradiction():
    """Same subject, attribute, period and scope, different magnitude."""
    a = {"subject": "India", "attribute": "GDP growth", "value": "6.4", "unit": "%",
         "time_scope": "FY25", "qualifier": None}
    b = {"subject": "India", "attribute": "GDP growth", "value": "7.2", "unit": "%",
         "time_scope": "FY25", "qualifier": None}
    verdict = deterministic_relation(a, b)
    assert verdict["relation_type"] == "CONTRADICTS"


def test_deterministic_relation_reconciles_a_period_difference():
    a = {"subject": "India", "attribute": "GDP growth", "value": "6.4", "unit": "%",
         "time_scope": "FY25"}
    b = {"subject": "India", "attribute": "GDP growth", "value": "7.2", "unit": "%",
         "time_scope": "FY24"}
    verdict = deterministic_relation(a, b)
    assert verdict["relation_type"] == "CONTEXT_RECONCILED"
    assert verdict["reason_tag"] == "different_period"


def test_deterministic_relation_corroborates_across_scales():
    """crore vs million normalise to the same magnitude, so this corroborates."""
    a = {"subject": "X", "attribute": "revenue", "value": "8,142", "unit": "INR crore"}
    b = {"subject": "X", "attribute": "revenue", "value": "81,420", "unit": "INR million"}
    assert deterministic_relation(a, b)["relation_type"] == "CORROBORATES"


def test_deterministic_relation_leaves_unrelated_facts_alone():
    a = {"subject": "X", "attribute": "revenue", "value": "100", "unit": "INR crore"}
    b = {"subject": "X", "attribute": "headcount", "value": "100", "unit": "people"}
    assert deterministic_relation(a, b)["relation_type"] == "UNRELATED"


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_backend_none_selects_the_deterministic_path():
    from fkl.llm.factory import build_client

    cfg = dataclasses.replace(CONFIG, llm_backend="none")
    assert build_client(cfg) is None


def test_unknown_backend_fails_loudly():
    from fkl.llm.factory import BackendUnavailable, build_client

    cfg = dataclasses.replace(CONFIG, llm_backend="wishful-thinking")
    with pytest.raises(BackendUnavailable):
        build_client(cfg)


def test_groq_without_a_key_raises_rather_than_silently_degrading(monkeypatch):
    """Someone who asked for Groq must be told it did not happen."""
    from fkl.llm.factory import build_client

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = dataclasses.replace(CONFIG, llm_backend="groq")
    with pytest.raises(Exception):
        build_client(cfg)


def test_describe_backend_is_honest_about_a_missing_key(monkeypatch):
    from fkl.llm.factory import describe_backend

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = dataclasses.replace(CONFIG, llm_backend="groq")
    assert "NO API KEY" in describe_backend(cfg)


# --------------------------------------------------------------------------- #
# Cloud fallback chain (LLM_BACKEND=groq+gemini)
# --------------------------------------------------------------------------- #


class _RateLimited(Exception):
    status_code = 429


class _BoomClient(LLMClient):
    """Raises a given exception on the first N calls, then answers."""

    def __init__(self, name, exc, fail_times):
        self.name = name
        self.model_name = name
        self.supports_json_mode = True
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    def complete(self, system, user, *, json_mode=False, max_tokens=768, temperature=0.0):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return LLMResponse(text="[]", model=self.model_name)


def test_fallback_switches_backend_once_the_first_is_exhausted():
    from fkl.llm.fallback_client import FallbackClient

    groq = _BoomClient("groq", _RateLimited(), fail_times=99)
    gemini = _BoomClient("gemini", _RateLimited(), fail_times=0)
    client = FallbackClient([groq, gemini])

    assert client.complete("s", "u").text == "[]"
    assert client.active is gemini
    # Groq is not retried on subsequent calls.
    client.complete("s", "u")
    assert groq.calls == 1 and gemini.calls == 2


def test_fallback_does_not_mask_a_non_quota_error():
    from fkl.llm.fallback_client import FallbackClient

    groq = _BoomClient("groq", ValueError("transient"), fail_times=1)
    gemini = _BoomClient("gemini", _RateLimited(), fail_times=0)
    client = FallbackClient([groq, gemini])

    with pytest.raises(ValueError):
        client.complete("s", "u")
    assert client.active is groq and gemini.calls == 0


def test_demo_force_groq_exhaust_raises_a_429_after_n_calls(monkeypatch):
    from fkl.llm.fallback_client import _is_exhaustion
    from fkl.llm.groq_client import GroqClient

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("FKL_DEMO_FORCE_GROQ_EXHAUST", "2")

    client = GroqClient.__new__(GroqClient)
    client.model_name = "m"
    client._demo_ok_left = 2
    client._client = None  # never reached: real calls happen only while _demo_ok_left > 0

    # First two calls fall through to the real path (None client -> AttributeError),
    # proving the demo guard did not fire; the third raises the synthetic 429.
    for _ in range(2):
        with pytest.raises(Exception) as ei:
            client.complete("s", "u")
        assert not _is_exhaustion(ei.value)
    with pytest.raises(Exception) as ei:
        client.complete("s", "u")
    assert _is_exhaustion(ei.value)


def test_chain_backend_builds_a_fallback_client(monkeypatch):
    from fkl.llm.factory import build_client
    from fkl.llm.fallback_client import FallbackClient

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = dataclasses.replace(CONFIG, llm_backend="groq+gemini")
    client = build_client(cfg)
    assert isinstance(client, FallbackClient)
