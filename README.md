# Fact Knowledge Layer

Extracts grounded facts from PDFs, links every fact to its exact evidence
(page number + verbatim quoted span), and identifies corroborations,
contradictions and context-reconciled relationships across documents.

Every stored fact carries a quote that is verified character-exact against the
page text the model actually saw. A fact whose quote cannot be located is
discarded rather than kept, so an invented number never reaches the store.

| Stage | Scope | Status |
|---|---|---|
| 1 | PDF parsing with page + character-offset grounding, chunking, SQLite schema, incremental ingest, CLI | ✅ done |
| 2 | LLM backends (llama.cpp / Groq / deterministic), JSON repair, fact extraction, evidence verification, dynamic schema registry | ✅ done |
| 3 | Embeddings + vector search, relationship classification, FastAPI, web UI, showcase view | ✅ done |

Measured on the six starter PDFs: 511 pages, 3,139 chunks, ingested in ~1.7 s.
**140 tests passing.**

## Setup and run

No accounts, no API keys and no background services are required. The default
local model is downloaded automatically on first run.

```bash
pip install -r requirements.txt

python run.py          # ingest -> extract -> link -> serve the web UI
```

Then open <http://localhost:8000/> for the UI (showcase, fact browser,
relationship browser, schema registry, upload) and <http://localhost:8000/docs>
for the API.

Every stage is resumable and skips work that is already done, so re-running is
cheap and interrupting with Ctrl-C is safe. A full-corpus extraction on CPU is
roughly an hour; use `python run.py --limit N` for a quick demo run, or
`python run.py --no-extract` to ingest and serve immediately.

Requires Python 3.10+ (tested through 3.13; 3.14 also works).

### Individual commands

```bash
python run.py ingest --dataset all     # parse every starter PDF
python run.py extract [--limit N]      # grounded fact extraction (resumable)
python run.py link                     # embed facts + classify relationships
python run.py facts | relations | schema
python run.py status                   # what's in the store
python run.py page <document> <page>   # verbatim page text + detected page label
python run.py serve                    # web UI + API only
python -m pytest tests -q
```

## Approach

**Grounding is the load-bearing invariant.** `pages.text` stores exactly what
PyMuPDF returned, and every chunk and every quote records character offsets into
that string. Matching a model's quote back to the source runs in tiers — exact,
whitespace-normalised, typography/case-folded, then a length-bounded
reconstructed span for column-major table text — and every tier returns offsets
into the original text. A quote that matches no tier means the fact is dropped.

**Extraction** processes one page-sized chunk at a time across worker processes,
each holding its own model; inference happens in the workers, while grounding and
every database write stay in the parent so SQLite remains single-writer. Output
is decoded under a grammar that constrains only the JSON envelope, so fact keys
stay free to emerge from the documents into a dynamic `fact_types` registry.
Malformed JSON goes through parse → corrective re-prompt → `json-repair` → give
up, with every outcome logged so the repair rate is reported from data.

**Backends** sit behind one interface: local `llama.cpp` + GGUF (default),
Groq (optional, only when `GROQ_API_KEY` is set), and a deterministic
pattern-based extractor that is the floor the system never falls through when no
model can load.

**Linking** embeds the *normalised* fact rather than the source sentence, does
brute-force exact nearest-neighbour search over vectors stored in the same SQLite
file, and only pairs facts from different documents. Classification is two-step:
a deterministic pass settles what arithmetic can decide (units, magnitudes,
periods) and hands that to the model as a hint. Relationships are stored as
first-class queryable rows — `CORROBORATES`, `CONTRADICTS`, `CONTEXT_RECONCILED`,
`UNRELATED` — each with both fact ids, a reason tag, an explanation, confidence
and similarity.

See **[DECISIONS.md](DECISIONS.md)** for the full build diary — the measurements,
the dead ends, and the bugs the real corpus exposed.

## Limitations and next steps

- Incremental ingest diffs at **document** granularity: a 100-page PDF with one
  changed page is reprocessed in full. Page-level hashing would fix it.
- All six starter PDFs are text-bearing; scanned documents are detected and
  reported but not OCR'd.
- Cross-currency conversion is deliberately not attempted — inventing an FX rate
  would manufacture false corroborations.
- Table detection is off by default (see Notes); financial-statement structure is
  read from the linearised page text instead.
- A full CPU extraction of the corpus is ~an hour; the practical demo path is
  `--limit` or the optional Groq backend.

## Notes

- `requirements.txt` opens with an `--extra-index-url` line that serves
  **prebuilt** CPU wheels for `llama-cpp-python`. Without it, pip compiles
  llama.cpp from source and needs `cmake` plus a C++ toolchain. With it, install
  is a plain wheel download.
- Embeddings use `fastembed` (ONNX) rather than `sentence-transformers`, to avoid
  a 1–2 GB `torch` dependency. Same models.
- Table detection is **off by default** — it costs ~500× plain text extraction on
  dense financial reports while producing worse output than the plain page text.
  Enable with `--tables`. Full reasoning and measurements in DECISIONS.md.
- GPU offload is automatic when a CUDA build of `llama-cpp-python` is installed;
  a normal clone installs the CPU wheel and nothing changes.
