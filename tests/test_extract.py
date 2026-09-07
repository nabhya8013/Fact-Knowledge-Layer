"""Stage-2 tests for the two pure-Python correctness guarantees:

* a quote is grounded to exact original offsets, or the fact is rejected;
* values are normalised so that differently-worded quantities become comparable.

Neither needs an LLM, so these run in milliseconds and gate the parts of the
pipeline where a silent bug would corrupt every downstream relationship.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fkl.extract.grounding import find_quote, locate_in_page  # noqa: E402
from fkl.extract.normalize import (  # noqa: E402
    canonical_key,
    canonical_text,
    fact_type_name,
    normalize_value,
    parse_number,
    values_agree,
)


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #

# Mirrors how PyMuPDF actually returns text: hard-wrapped mid-sentence.
WRAPPED = (
    "Card payments \nincreased by 17.9 per cent in value terms and 35 \n"
    "per cent in volume terms during 2024-25.\n"
)


def test_exact_quote_is_found_verbatim():
    m = find_quote(WRAPPED, "per cent in volume terms during 2024-25")
    assert m is not None and m.match_mode == "exact"
    assert WRAPPED[m.start : m.end] == "per cent in volume terms during 2024-25"


def test_line_wrapped_quote_still_grounds_to_exact_offsets():
    """The critical case: the model writes a space where the PDF has a newline."""
    quote = "increased by 17.9 per cent in value terms and 35 per cent"
    m = find_quote(WRAPPED, quote)
    assert m is not None
    assert m.match_mode == "normalised_whitespace"
    # Offsets must address the ORIGINAL text, newline included.
    assert WRAPPED[m.start : m.end] == "increased by 17.9 per cent in value terms and 35 \nper cent"


def test_typographic_and_case_differences_are_forgiven():
    source = "Revenue rose to ₹8,142 crore — an increase of 12%."
    m = find_quote(source, "revenue rose to ₹8,142 crore - an increase of 12%")
    assert m is not None and m.match_mode == "normalised_chars"
    assert source[m.start : m.end].startswith("Revenue rose to")


def test_hallucinated_value_is_rejected():
    """A number the source does not contain must not be groundable."""
    assert find_quote(WRAPPED, "increased by 99.9 per cent in value terms") is None


def test_short_quotes_are_rejected():
    """Short strings match by chance and are useless as evidence."""
    assert find_quote(WRAPPED, "35") is None
    assert find_quote(WRAPPED, "per cent") is None


def test_model_added_quotation_marks_are_stripped():
    m = find_quote(WRAPPED, '"per cent in volume terms during 2024-25"')
    assert m is not None


# PyMuPDF emits table text column-major, so a row label, its value and the
# column's unit header are never contiguous.
TABLE_TEXT = """Particulars
As at March 31, 2024
As at March 31, 2023
(INR million)
Total equity
59,798.47
29,148.37
Non-current liabilities
Borrowings
1,005.28
1,329.84
"""


def test_table_fact_grounds_via_reconstructed_span():
    """The model assembles "Total equity 59,798.47 INR million" from three
    separate places on the page. That is a real fact and must not be lost."""
    from fkl.extract.grounding import find_reconstructed_span

    assert find_quote(TABLE_TEXT, "Total equity 59,798.47 INR million") is None
    m = find_reconstructed_span(TABLE_TEXT, "Total equity 59,798.47 INR million")
    assert m is not None and m.match_mode == "reconstructed_span"
    # Evidence must be REAL source text, never the model's reconstruction.
    assert m.text == TABLE_TEXT[m.start : m.end]
    assert "59,798.47" in m.text


def test_reconstructed_span_still_rejects_hallucinated_numbers():
    """The relaxed tier must not become a licence to invent figures."""
    from fkl.extract.grounding import find_reconstructed_span

    assert find_reconstructed_span(TABLE_TEXT, "Total equity 99,999.99 INR million") is None
    assert find_reconstructed_span(TABLE_TEXT, "Total equity 12,345.67 INR million") is None


def test_reconstructed_span_is_bounded():
    """Tokens scattered far apart must not be stitched into one 'fact'."""
    from fkl.extract.grounding import find_reconstructed_span

    far = "Total equity\n59,798.47\n" + ("filler text line\n" * 60) + "Borrowings\n1,005.28\n"
    assert find_reconstructed_span(far, "Total equity 59,798.47 Borrowings 1,005.28") is None


def test_locate_in_page_translates_chunk_offsets_to_page_offsets():
    page = "HEADER\n" + WRAPPED
    chunk_start = len("HEADER\n")
    chunk = WRAPPED
    quote = "increased by 17.9 per cent in value terms and 35 per cent"
    m = locate_in_page(page, chunk, chunk_start, quote)
    assert m is not None
    assert page[m.start : m.end].startswith("increased by 17.9")


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("8142", 8142.0),          # regression: ordered alternation truncated this to 814
        ("2024", 2024.0),          # regression: and this to 202
        ("81.42", 81.42),
        ("1,234,567", 1234567.0),  # western grouping
        ("12,34,567", 1234567.0),  # indian grouping
        ("8,142", 8142.0),
        (".5", 0.5),
        ("(1,234)", -1234.0),      # accounting negative
        ("no digits here", None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        ("Rs 8,142 crore", None, (81_420_000_000.0, "INR")),
        ("8142", "₹ crore", (81_420_000_000.0, "INR")),
        ("INR 81.42 billion", None, (81_420_000_000.0, "INR")),
        ("$1.2 billion", None, (1_200_000_000.0, "USD")),
        ("17.9", "per cent", (17.9, "%")),
        ("6.4%", None, (6.4, "%")),
        ("45", "bps", (45.0, "bps")),
        ("1.2 million", "shipments", (1_200_000.0, "shipments")),
        ("market leader", None, (None, None)),
    ],
)
def test_normalize_value(value, unit, expected):
    assert normalize_value(value, unit) == expected


@pytest.mark.parametrize(
    "value",
    [
        "U63090DL2011PLC221234",                              # corporate identity number
        "N24-N34, S24-S34, Air Cargo Logistics Centre-II",    # address
        "Plot 5, Sector 44, Gurugram 122002 Haryana, India",  # address
        "Sunil Kumar Bansal",
        "www.delhivery.com",
    ],
)
def test_identifiers_and_addresses_do_not_yield_a_magnitude(value):
    """Regression: these produced 63090.0, 24.0 and 5.0 on the real corpus.

    A fabricated magnitude is worse than none, because linking would compare it
    against genuine figures and invent contradictions out of postcodes.
    """
    number, _ = normalize_value(value)
    assert number is None


def test_word_only_quotes_do_not_use_the_reconstructed_tier():
    """Regression: without numeric anchors this grounded 'registered office' to a
    200-character blob spanning six unrelated headers on a prospectus cover."""
    from fkl.extract.grounding import find_reconstructed_span

    source = "CORPORATE IDENTITY NUMBER\nREGISTERED OFFICE\nCORPORATE\nOFFICE\nCONTACT\nPERSON\n"
    assert find_reconstructed_span(source, "registered office corporate office contact person") is None


def test_crore_and_billion_are_recognised_as_the_same_quantity():
    """This is what makes the 'units differ' reconciliation case detectable."""
    a, unit_a = normalize_value("Rs 8,142 crore")
    b, unit_b = normalize_value("INR 81.42 billion")
    assert unit_a == unit_b == "INR"
    assert values_agree(a, b) is True


def test_different_currencies_are_not_silently_compared():
    """No FX rate is invented, so cross-currency facts stay incomparable."""
    _, unit_a = normalize_value("Rs 100 crore")
    _, unit_b = normalize_value("$100 million")
    assert unit_a != unit_b


def test_values_agree_distinguishes_unknown_from_disagreement():
    assert values_agree(None, 5.0) is None
    assert values_agree(5.0, 5.0) is True
    assert values_agree(5.0, 50.0) is False
    assert values_agree(100.0, 100.5, tolerance=0.01) is True


# --------------------------------------------------------------------------- #
# Identity / dynamic schema
# --------------------------------------------------------------------------- #


def test_canonical_key_ignores_cosmetic_differences():
    assert canonical_key("The Delhivery Limited", "Total Revenue") == canonical_key(
        "delhivery limited", "revenue"
    )


def test_canonical_text_is_built_from_fields_not_raw_sentence():
    text = canonical_text(
        {"subject": "Delhivery", "attribute": "revenue", "value": "8142",
         "unit": "INR crore", "time_scope": "FY24"}
    )
    assert "Delhivery" in text and "revenue" in text and "8142" in text and "FY24" in text


def test_fact_type_name_drives_the_dynamic_registry():
    """New attribute strings become new fact types with no migration."""
    assert fact_type_name({"attribute": "Real GDP growth"}) == "real_gdp_growth"
    assert fact_type_name({"predicate": "revenue"}) == "revenue"
    assert fact_type_name({}) == "unspecified"
