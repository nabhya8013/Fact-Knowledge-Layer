"""Stage-3 tests: showcase selection, API contracts and evidence verification.

Built on a synthetic two-document store with known relationships, so the
assertions are about the *system's* behaviour rather than about whatever the
model happened to extract on a given run. The real corpus is exercised
separately by the end-to-end pipeline test.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.config import CONFIG  # noqa: E402
from fkl.db import connect  # noqa: E402
from fkl.showcase import build_showcase, failure_report  # noqa: E402

# Two documents whose page text really contains the quotes below, so evidence
# offsets can be verified rather than asserted by construction.
PAGE_A = (
    "Delhivery Limited reported revenue from operations of Rs 8,142 crore in FY24,\n"
    "compared with Rs 6,882 crore in FY23. Real GDP growth was 6.4 per cent in FY25.\n"
)
PAGE_B = (
    "Revenue from contract with customers stood at 81,420 INR million for FY24.\n"
    "Real GDP growth is projected at 7.2 per cent in FY25 by the staff report.\n"
)


def _insert_document(conn, doc_id, filename, page_text):
    conn.execute(
        """INSERT INTO documents (id, content_sha256, filename, title, page_count,
                                  byte_size, parser_version, status, ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_id, doc_id.ljust(64, "0"), filename, filename, 1, 100, "1", "extracted", "2026-01-01"),
    )
    conn.execute(
        """INSERT INTO pages (id, document_id, pdf_page_index, printed_page_label,
                              text, char_count) VALUES (?,?,?,?,?,?)""",
        (f"{doc_id}:p0", doc_id, 0, "12", page_text, len(page_text)),
    )
    conn.execute(
        """INSERT INTO chunks (id, document_id, page_id, pdf_page_index, printed_page_label,
                               role, kind, ordinal, char_start, char_end, text,
                               token_estimate, numeric_density)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"{doc_id}:e0", doc_id, f"{doc_id}:p0", 0, "12", "extraction", "prose",
         0, 0, len(page_text), page_text, 60, 0.4),
    )


def _insert_fact(conn, fact_id, doc_id, page_text, payload, quote, value_num, unit):
    conn.execute(
        """INSERT INTO facts (id, document_id, chunk_id, payload_json, fact_type,
                              canonical_key, canonical_text, value_num, unit, time_scope,
                              confidence, extractor, embedded, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
        (fact_id, doc_id, f"{doc_id}:e0", json.dumps(payload), payload["attribute"],
         f"{payload['subject']}::{payload['attribute']}",
         f"{payload['subject']} - {payload['attribute']}: {payload['value']}",
         value_num, unit, payload.get("time_scope"), 0.9, "local-llm", "2026-01-01"),
    )
    start = page_text.index(quote)
    conn.execute(
        """INSERT INTO evidence (id, fact_id, document_id, chunk_id, pdf_page_index,
                                 printed_page_label, quote, quote_char_start,
                                 quote_char_end, match_mode)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (f"ev_{fact_id}", fact_id, doc_id, f"{doc_id}:e0", 0, "12", quote,
         start, start + len(quote), "exact"),
    )


def _insert_relationship(conn, rel_id, a, b, rtype, reason, explanation):
    conn.execute(
        """INSERT INTO relationships (id, fact_a_id, fact_b_id, relation_type, reason_tag,
                                      explanation, confidence, similarity_score,
                                      cross_document, model, created_at)
           VALUES (?,?,?,?,?,?,?,?,1,?,?)""",
        (rel_id, a, b, rtype, reason, explanation, 0.9, 0.95, "test-model", "2026-01-01"),
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    cfg = dataclasses.replace(CONFIG, data_dir=tmp_path)
    monkeypatch.setattr("fkl.config.CONFIG", cfg)
    monkeypatch.setattr("fkl.api.CONFIG", cfg)

    conn = connect(cfg.db_path)
    _insert_document(conn, "doc_aaa", "annual-report.pdf", PAGE_A)
    _insert_document(conn, "doc_bbb", "staff-report.pdf", PAGE_B)

    _insert_fact(conn, "fact_rev_a", "doc_aaa", PAGE_A,
                 {"subject": "Delhivery", "attribute": "revenue", "value": "8,142",
                  "unit": "INR crore", "time_scope": "FY24"},
                 "revenue from operations of Rs 8,142 crore in FY24", 8.142e10, "INR")
    _insert_fact(conn, "fact_rev_b", "doc_bbb", PAGE_B,
                 {"subject": "Delhivery", "attribute": "revenue from contract with customers",
                  "value": "81,420", "unit": "INR million", "time_scope": "FY24"},
                 "81,420 INR million for FY24", 8.142e10, "INR")
    _insert_fact(conn, "fact_gdp_a", "doc_aaa", PAGE_A,
                 {"subject": "India", "attribute": "real GDP growth", "value": "6.4",
                  "unit": "%", "time_scope": "FY25"},
                 "Real GDP growth was 6.4 per cent in FY25", 6.4, "%")
    _insert_fact(conn, "fact_gdp_b", "doc_bbb", PAGE_B,
                 {"subject": "India", "attribute": "real GDP growth", "value": "7.2",
                  "unit": "%", "time_scope": "FY25"},
                 "projected at 7.2 per cent in FY25", 7.2, "%")
    _insert_fact(conn, "fact_rev_prior", "doc_aaa", PAGE_A,
                 {"subject": "Delhivery", "attribute": "revenue", "value": "6,882",
                  "unit": "INR crore", "time_scope": "FY23"},
                 "Rs 6,882 crore in FY23", 6.882e10, "INR")

    _insert_relationship(conn, "rel_1", "fact_rev_a", "fact_rev_b", "CORROBORATES",
                         "units_differ",
                         "Both state FY24 revenue; 8,142 crore equals 81,420 million.")
    _insert_relationship(conn, "rel_2", "fact_gdp_a", "fact_gdp_b", "CONTRADICTS",
                         "value_mismatch",
                         "Same metric, same period FY25, but 6.4% versus 7.2%.")
    _insert_relationship(conn, "rel_3", "fact_rev_prior", "fact_rev_b", "CONTEXT_RECONCILED",
                         "different_period",
                         "Values differ because the periods differ: FY23 versus FY24.")

    conn.execute(
        """INSERT INTO extraction_progress (chunk_id, status, facts, processed_at)
           VALUES ('doc_aaa:e0','done',3,'2026'), ('doc_bbb:e0','done',2,'2026')"""
    )
    # Only chunks whose JSON needed help are logged; a clean first-try parse is
    # not. So one of the two attempted chunks (doc_bbb) needed a re-prompt.
    conn.execute(
        """INSERT INTO repair_log (chunk_id, stage, attempts, outcome, created_at)
           VALUES ('doc_bbb:e0','extraction',2,'ok_after_reprompt','2026')"""
    )
    conn.commit()
    yield conn, cfg
    conn.close()


@pytest.fixture
def client(store):
    from fastapi.testclient import TestClient

    from fkl.api import app

    return TestClient(app)


# --------------------------------------------------------------------------- #
# Showcase
# --------------------------------------------------------------------------- #


def test_showcase_finds_all_three_relationship_cases(store):
    conn, _ = store
    result = build_showcase(conn)
    assert result["complete"], f"missing cases: {result['missing']}"
    for key in ("corroboration", "contradiction", "context_reconciled"):
        assert result["cases"][key]["examples"], f"no example for {key}"


def test_showcase_examples_carry_evidence_from_both_documents(store):
    """A showcase example is only useful if the reviewer can check both sides."""
    conn, _ = store
    result = build_showcase(conn)
    for key in ("corroboration", "contradiction", "context_reconciled"):
        example = result["cases"][key]["examples"][0]
        a, b = example["fact_a"], example["fact_b"]
        assert a["document_id"] != b["document_id"], "must be cross-document"
        for side in (a, b):
            assert side["evidence"]["quote"], "each side needs a quote"
            assert side["evidence"]["page_label"]
        assert example["explanation"].strip()


def test_showcase_reports_incompleteness_rather_than_faking_it(store):
    """With no contradictions present, the showcase must say so."""
    conn, _ = store
    conn.execute("DELETE FROM relationships WHERE relation_type = 'CONTRADICTS'")
    conn.commit()
    result = build_showcase(conn)
    assert not result["complete"]
    assert "contradiction" in result["missing"]
    assert result["cases"]["contradiction"]["examples"] == []


def test_failure_report_is_computed_from_real_counters(store):
    conn, _ = store
    report = failure_report(conn)
    assert report["chunks_attempted"] == 2
    assert report["json_calls_needing_repair"] == 1
    # One of two attempted chunks needed a re-prompt; none failed outright.
    assert report["json_repair_rate"] == pytest.approx(0.5)
    assert report["json_failure_rate"] == pytest.approx(0.0)
    assert report["evidence_match_modes"] == {"exact": 5}


def test_failure_report_repair_rate_is_not_trivially_one(store):
    """Regression: the rate divided repaired calls by themselves and was always 1.0."""
    conn, _ = store
    # A second clean chunk on doc_aaa: attempted, never logged to repair_log.
    conn.execute(
        """INSERT INTO chunks (id, document_id, page_id, pdf_page_index, printed_page_label,
                               role, kind, ordinal, char_start, char_end, text,
                               token_estimate, numeric_density)
           VALUES ('doc_aaa:e1','doc_aaa','doc_aaa:p0',0,'12','extraction','prose',
                   1,0,10,'more text',10,0.1)"""
    )
    conn.execute(
        "INSERT INTO extraction_progress (chunk_id, status, facts, processed_at) "
        "VALUES ('doc_aaa:e1','done',1,'2026')"
    )
    conn.commit()
    assert failure_report(conn)["json_repair_rate"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def test_health_reports_counts(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["counts"]["facts"] == 5
    assert body["counts"]["relationships"] == 3


def test_documents_listing(client):
    docs = client.get("/api/documents").json()
    assert {d["filename"] for d in docs} == {"annual-report.pdf", "staff-report.pdf"}
    assert all("meta" in d for d in docs)


def test_facts_endpoint_filters(client):
    assert client.get("/api/facts").json()["total"] == 5
    assert client.get("/api/facts?document_id=doc_aaa").json()["total"] == 3
    assert client.get("/api/facts?numeric_only=true").json()["total"] == 5
    assert client.get("/api/facts?q=GDP").json()["total"] == 2
    assert client.get("/api/facts?min_confidence=0.95").json()["total"] == 0


def test_fact_detail_includes_its_relationships(client):
    body = client.get("/api/facts/fact_gdp_a").json()
    assert body["subject"] == "India"
    types = {r["relation_type"] for r in body["relationships"]}
    assert "CONTRADICTS" in types
    other = body["relationships"][0]["other_fact"]
    assert other["document_id"] == "doc_bbb"


def test_relationships_endpoint_excludes_unrelated_by_default(client, store):
    conn, _ = store
    _insert_relationship(conn, "rel_4", "fact_rev_a", "fact_gdp_b", "UNRELATED",
                         "different_attribute", "Different properties entirely.")
    conn.commit()

    assert client.get("/api/relationships").json()["total"] == 3
    assert client.get("/api/relationships?include_unrelated=true").json()["total"] == 4
    assert client.get("/api/relationships?relation_type=CONTRADICTS").json()["total"] == 1


def test_fact_types_registry_is_exposed(client):
    rows = client.get("/api/fact-types").json()
    assert rows == [] or all("observed_keys" in r for r in rows)


def test_evidence_context_offsets_are_verified(client):
    """The reviewer-facing guarantee: the stored offsets address the stored quote."""
    body = client.get("/api/evidence/fact_rev_a/context").json()
    assert body["offsets_verified"] is True
    assert body["highlight"] == "revenue from operations of Rs 8,142 crore in FY24"
    assert "Delhivery Limited reported" in body["before"]
    assert body["printed_page_label"] == "12"


def test_evidence_context_is_reconstructable_from_offsets(client, store):
    """before + highlight + after must be a contiguous slice of the real page."""
    conn, _ = store
    body = client.get("/api/evidence/fact_gdp_a/context").json()
    page_text = conn.execute(
        "SELECT text FROM pages WHERE id = 'doc_aaa:p0'"
    ).fetchone()["text"]
    assert body["before"] + body["highlight"] + body["after"] in page_text


def test_showcase_endpoint_matches_the_module(client):
    body = client.get("/api/showcase").json()
    assert body["complete"] is True
    assert body["totals"]["facts"] == 5


def test_unknown_ids_return_404(client):
    assert client.get("/api/facts/fact_nope").status_code == 404
    assert client.get("/api/evidence/fact_nope/context").status_code == 404
    assert client.get("/api/jobs/job_nope").status_code == 404


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Fact Knowledge Layer" in r.text
