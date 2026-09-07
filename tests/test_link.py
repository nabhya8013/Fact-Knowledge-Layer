"""Tests for the linking layer: candidate generation and pair classification.

The vector maths is exercised through the real `VectorIndex`; the LLM half uses
a scripted client so the classifier's own logic - especially the deterministic
veto over a wrongly-claimed contradiction - is tested without a model.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.config import CONFIG  # noqa: E402
from fkl.db import connect  # noqa: E402
from fkl.link.relate import (  # noqa: E402
    _contradiction_blocker,
    classify_pair,
    link_facts,
    load_fact_view,
)
from fkl.link.vector_index import Candidate, VectorIndex  # noqa: E402
from fkl.llm.base import LLMClient, LLMResponse  # noqa: E402


class ScriptedClient(LLMClient):
    name = "scripted"
    model_name = "scripted"
    supports_json_mode = True

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, user, *, json_mode=False, max_tokens=768, temperature=0.0):
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, model=self.model_name)


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


def _unit_index(rows):
    """rows: list of (fact_id, document_id, vector). Vectors are L2-normalised."""
    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    mat = np.array([r[2] for r in rows], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return VectorIndex(ids, docs, mat)


def test_cross_document_pairs_only_pairs_different_documents():
    idx = _unit_index([
        ("fact_a", "doc_1", [1.0, 0.0]),
        ("fact_b", "doc_1", [1.0, 0.01]),   # near-identical, but same document
        ("fact_c", "doc_2", [1.0, 0.0]),    # identical to fact_a, different document
    ])
    pairs = idx.cross_document_pairs(threshold=0.9)
    got = {tuple(sorted((p.fact_a_id, p.fact_b_id))) for p in pairs}
    assert ("fact_a", "fact_c") in got
    assert ("fact_a", "fact_b") not in got  # same document must never pair


def test_cross_document_pairs_emits_each_pair_once():
    idx = _unit_index([
        ("fact_a", "doc_1", [1.0, 0.0]),
        ("fact_c", "doc_2", [1.0, 0.0]),
    ])
    pairs = idx.cross_document_pairs(threshold=0.5)
    assert len(pairs) == 1
    assert tuple(sorted((pairs[0].fact_a_id, pairs[0].fact_b_id))) == ("fact_a", "fact_c")


def test_cross_document_pairs_respects_threshold():
    idx = _unit_index([
        ("fact_a", "doc_1", [1.0, 0.0]),
        ("fact_c", "doc_2", [0.0, 1.0]),   # orthogonal -> similarity 0
    ])
    assert idx.cross_document_pairs(threshold=0.5) == []


# --------------------------------------------------------------------------- #
# The deterministic veto over a wrongly-claimed contradiction
# --------------------------------------------------------------------------- #

REV_FY24 = {"subject": "Acme", "attribute": "revenue", "value": "100",
            "unit": "INR crore", "time_scope": "FY24"}
REV_FY23 = {"subject": "Acme", "attribute": "revenue", "value": "80",
            "unit": "INR crore", "time_scope": "FY23"}
REV_FY24_MN = {"subject": "Acme", "attribute": "revenue", "value": "1000",
               "unit": "USD million", "time_scope": "FY24"}
REV_FY24_B = {"subject": "Acme", "attribute": "revenue", "value": "120",
              "unit": "INR crore", "time_scope": "FY24"}


def test_contradiction_blocker_flags_period_and_unit_mismatch():
    assert _contradiction_blocker(REV_FY24, REV_FY23)[0] == "different_period"
    assert _contradiction_blocker(REV_FY24, REV_FY24_MN)[0] == "units_differ"


def test_contradiction_blocker_allows_a_genuine_conflict():
    # Same period, same (INR) unit, different value: nothing blocks a contradiction.
    assert _contradiction_blocker(REV_FY24, REV_FY24_B) is None


_CONTRADICTS = json.dumps([{
    "relation_type": "CONTRADICTS", "reason_tag": "different_value",
    "explanation": "Revenue values differ.", "confidence": 0.9,
}])


def test_classify_pair_downgrades_contradiction_on_period_mismatch():
    verdict, _ = classify_pair(REV_FY24, REV_FY23, ScriptedClient([_CONTRADICTS]), CONFIG)
    assert verdict["relation_type"] == "CONTEXT_RECONCILED"
    assert verdict["reason_tag"] == "different_period"
    assert "FY24" in verdict["explanation"] and "FY23" in verdict["explanation"]


def test_classify_pair_keeps_a_genuine_contradiction():
    verdict, _ = classify_pair(REV_FY24, REV_FY24_B, ScriptedClient([_CONTRADICTS]), CONFIG)
    assert verdict["relation_type"] == "CONTRADICTS"


def test_classify_pair_falls_back_to_rules_without_a_model():
    verdict, outcome = classify_pair(REV_FY24, REV_FY23, None, CONFIG)
    assert verdict["relation_type"] in {"CONTEXT_RECONCILED", "UNRELATED", "CONTRADICTS"}
    assert outcome == "ok_first_try"


# --------------------------------------------------------------------------- #
# link_facts: incremental, skip-existing
# --------------------------------------------------------------------------- #


@pytest.fixture
def linked_store(tmp_path):
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path, llm_backend="none",
                              similarity_threshold=0.5)
    conn = connect(cfg.db_path)

    for doc_id, fname in (("doc_a0", "a.pdf"), ("doc_b0", "b.pdf")):
        conn.execute(
            """INSERT INTO documents (id, content_sha256, filename, title, page_count,
                                      byte_size, parser_version, status, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (doc_id, doc_id.ljust(64, "0"), fname, fname, 1, 100, "1", "extracted", "2026-01-01"),
        )
        conn.execute(
            """INSERT INTO pages (id, document_id, pdf_page_index, printed_page_label,
                                  text, char_count) VALUES (?,?,?,?,?,?)""",
            (f"{doc_id}:p0", doc_id, 0, "1", "x" * 50, 50),
        )
        conn.execute(
            """INSERT INTO chunks (id, document_id, page_id, pdf_page_index,
                                   printed_page_label, role, kind, ordinal, char_start,
                                   char_end, text, token_estimate, numeric_density)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"{doc_id}:e0", doc_id, f"{doc_id}:p0", 0, "1", "extraction", "prose",
             0, 0, 50, "x" * 50, 12, 0.5),
        )

    def add_fact(fact_id, doc_id, payload, vector):
        conn.execute(
            """INSERT INTO facts (id, document_id, chunk_id, payload_json, fact_type,
                                  canonical_key, canonical_text, value_num, unit,
                                  time_scope, confidence, extractor, embedded, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (fact_id, doc_id, f"{doc_id}:e0", json.dumps(payload), payload["attribute"],
             f"{payload['subject']}::{payload['attribute']}", payload["value"],
             float(payload["value"]), "INR", payload.get("time_scope"), 0.9,
             "local-llm", "2026-01-01"),
        )
        v = np.asarray(vector, dtype=np.float32)
        v /= np.linalg.norm(v)
        conn.execute(
            "INSERT INTO fact_embeddings (fact_id, dim, model, vector) VALUES (?,?,?,?)",
            (fact_id, len(v), "test", v.tobytes()),
        )

    add_fact("fact_a1", "doc_a0",
             {"subject": "Acme", "attribute": "revenue", "value": "100", "time_scope": "FY24"},
             [1.0, 0.0])
    add_fact("fact_b1", "doc_b0",
             {"subject": "Acme", "attribute": "revenue", "value": "100", "time_scope": "FY24"},
             [1.0, 0.0])
    conn.commit()
    yield conn, cfg
    conn.close()


def test_link_facts_classifies_the_cross_document_pair(linked_store):
    conn, cfg = linked_store
    stats = link_facts(conn, cfg, None)
    assert stats.candidates == 1
    assert stats.stored == 1
    row = conn.execute("SELECT * FROM relationships").fetchone()
    assert (row["fact_a_id"], row["fact_b_id"]) == ("fact_a1", "fact_b1")


def test_link_facts_skips_a_pair_already_classified(linked_store):
    conn, cfg = linked_store
    link_facts(conn, cfg, None)
    stats = link_facts(conn, cfg, None)  # second run
    assert stats.candidates == 1
    assert stats.skipped_existing == 1
    assert stats.stored == 0
    assert conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 1


def test_load_fact_view_carries_payload_and_evidence(linked_store):
    conn, _ = linked_store
    view = load_fact_view(conn, "fact_a1")
    assert view["subject"] == "Acme" and view["attribute"] == "revenue"
    assert view["time_scope"] == "FY24"
