"""Prompts for fact extraction.

Two constraints shaped every word here:

1. **Nothing may be document-specific.** The worked example below is deliberately
   synthetic - a fictional company and a fictional country - so the model is
   shown the *shape* of a fact without being primed with any entity, metric or
   phrasing from the starter PDFs. Using a real example from the corpus would be
   a subtle form of hardcoding: it would lift scores on these documents and
   quietly fail on unseen ones.

2. **The key set is suggested, not mandated.** The prompt names a core set of
   fields and then explicitly invites extra keys where the text supports them.
   That is what lets the stored schema grow to fit whatever the documents
   actually contain, rather than forcing every document into a fixed shape.
"""

from __future__ import annotations

EXTRACTION_SYSTEM = """\
You extract verifiable facts from document excerpts.

Return ONLY a JSON array. Each element is one fact, an object with these keys:
  subject      - the entity the fact is about (company, country, institution)
  attribute    - the property being stated (e.g. "revenue", "headcount",
                 "real GDP growth", "date of incorporation")
  value        - the stated value, as written in the text
  unit         - the unit or currency as written (e.g. "INR crore", "%", "people").
                 Use null if the value has no unit.
  time_scope   - the period or point in time the fact applies to, as written
                 (e.g. "FY24", "Q4 FY24", "as of 31 March 2024"). null if absent.
  qualifier    - any scope limitation that changes the meaning, e.g. "consolidated",
                 "standalone", "excluding one-time items", "projected", "revised".
                 null if absent.
  confidence   - your confidence from 0.0 to 1.0 that this fact is correctly read.
  source_quote - the EXACT span of the provided text that states this fact,
                 copied character for character.

You may add extra keys when the text supports them (for example "segment",
"region", "counterparty", "basis"). Do not invent keys that the text does not
justify.

Rules:
- source_quote MUST be copied verbatim from the text given to you. Never
  paraphrase, never summarise, never fix typos, never merge separated sentences.
- Extract only what the text states. Never infer, compute, or bring in outside
  knowledge.
- Prefer specific, checkable facts (numbers, dates, named quantities) over vague
  statements.
- If the text contains no verifiable facts, return [].
"""

# Synthetic on purpose: no entity, metric or phrasing from the real corpus.
_EXAMPLE_TEXT = """\
Northwind Freight Ltd reported consolidated revenue of Rs 1,240 crore
in FY23, up from Rs 980 crore a year earlier. The company operated
42 distribution centres as of 31 March 2023."""

_EXAMPLE_JSON = """\
[
  {"subject": "Northwind Freight Ltd", "attribute": "revenue", "value": "1,240",
   "unit": "INR crore", "time_scope": "FY23", "qualifier": "consolidated",
   "confidence": 0.95,
   "source_quote": "consolidated revenue of Rs 1,240 crore\\nin FY23"},
  {"subject": "Northwind Freight Ltd", "attribute": "revenue", "value": "980",
   "unit": "INR crore", "time_scope": "FY22", "qualifier": "consolidated",
   "confidence": 0.8,
   "source_quote": "up from Rs 980 crore a year earlier"},
  {"subject": "Northwind Freight Ltd", "attribute": "distribution centres",
   "value": "42", "unit": "centres", "time_scope": "as of 31 March 2023",
   "qualifier": null, "confidence": 0.95,
   "source_quote": "operated\\n42 distribution centres as of 31 March 2023"}
]"""


def build_extraction_prompt(chunk_text: str, *, document_title: str | None = None) -> str:
    """User-turn prompt for one chunk.

    The document title is passed as weak context only - it helps the model
    resolve pronouns and unnamed subjects ("the Company") - and the instruction
    explicitly forbids treating it as a source of facts in its own right.
    """
    header = ""
    if document_title:
        header = (
            f"The excerpt below is from a document titled: {document_title}\n"
            "Use that only to resolve who 'the Company' or 'the Bank' refers to. "
            "Do not extract facts from the title itself.\n\n"
        )

    return (
        f"{header}"
        "EXAMPLE\n"
        f"Text:\n{_EXAMPLE_TEXT}\n\n"
        f"Facts:\n{_EXAMPLE_JSON}\n\n"
        "NOW DO THE SAME FOR THIS TEXT.\n"
        f"Text:\n{chunk_text}\n\n"
        "Facts (JSON array only):"
    )


# --------------------------------------------------------------------------- #
# Relationship classification (used in stage 3, defined here alongside its twin)
# --------------------------------------------------------------------------- #

RELATION_SYSTEM = """\
You compare two facts extracted from different documents and classify their
relationship.

Return ONLY a JSON array containing exactly one object, with these keys:
  relation_type - one of:
      CORROBORATES       the two facts assert the same thing, however differently
                         worded or differently scaled
      CONTRADICTS        the two facts genuinely conflict: same subject, same
                         attribute, same period and scope, incompatible values
      CONTEXT_RECONCILED the values differ but the difference is fully explained
                         by context - different time period, different scope
                         (consolidated vs standalone, forecast vs actual),
                         different units, or a different basis of measurement
      UNRELATED          the facts are about different things and do not bear on
                         each other
  reason_tag    - a short snake_case tag for the specific reason, e.g.
                  different_period, different_scope, units_differ, rephrasing,
                  forecast_vs_actual, different_entity, revision
  explanation   - one or two sentences naming the concrete reason, citing the
                  specific difference. Be precise: say "FY23 versus FY24", not
                  "different periods".
  confidence    - 0.0 to 1.0

Judge only what the two facts and their quotes state. Do not use outside
knowledge. If the periods or scopes differ at all, that is CONTEXT_RECONCILED,
not CONTRADICTS - a contradiction requires that everything else match.
"""


def build_relation_prompt(fact_a: dict, fact_b: dict) -> str:
    """User-turn prompt comparing two facts, each shown with its evidence."""

    def render(label: str, fact: dict) -> str:
        return (
            f"{label}:\n"
            f"  document:   {fact.get('document_title') or fact.get('document_id')}\n"
            f"  subject:    {fact.get('subject')}\n"
            f"  attribute:  {fact.get('attribute')}\n"
            f"  value:      {fact.get('value')} {fact.get('unit') or ''}\n"
            f"  time_scope: {fact.get('time_scope')}\n"
            f"  qualifier:  {fact.get('qualifier')}\n"
            f"  quote:      \"{fact.get('source_quote')}\"\n"
        )

    return (
        f"{render('FACT A', fact_a)}\n"
        f"{render('FACT B', fact_b)}\n"
        "Classify the relationship (JSON array with one object):"
    )
