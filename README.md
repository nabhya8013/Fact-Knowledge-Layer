# Fact Knowledge Layer

Extracts grounded facts from PDFs, links every fact to its exact evidence
(page number + verbatim quoted span), and identifies where facts **corroborate**,
**contradict**, or can be **reconciled through context** across documents.

Every stored fact carries a quote that is verified character-exact against the
page text the model actually saw. A fact whose quote cannot be located is
discarded rather than kept, so an invented number never reaches the store.

Upload new PDFs through the web UI or the API — nothing in the pipeline is
specific to the starter files. No hard-coded facts, filenames, or schemas: the
fact shape emerges from the documents into a dynamic registry.

| Stage | Scope | Status |
|---|---|---|
| 1 | PDF parsing with page + character-offset grounding, chunking, SQLite schema, incremental ingest, CLI | ✅ done |
| 2 | LLM backends (llama.cpp / Gemini / Groq / deterministic), JSON repair, fact extraction, evidence verification, dynamic schema registry | ✅ done |
| 3 | Embeddings + vector search, relationship classification, FastAPI, web UI, showcase view | ✅ done |

Measured on the six starter PDFs: 511 pages, 3,139 chunks, ingested in ~1.7 s.
**154 tests passing.**

---

## Setup and run instructions

No accounts, no API keys and no background services are required. The default
local model (`Qwen2.5-1.5B-Instruct` GGUF, ~1 GB) downloads automatically on the
first run.

```bash
pip install -r requirements.txt

python run.py          # ingest -> extract -> link -> serve the web UI
```

Then open:

- <http://localhost:8000/> — the web UI: **Showcase** (the four required cases),
  fact browser with confidence filter, relationship browser, the dynamic schema
  registry, and PDF upload with live job progress.
- <http://localhost:8000/docs> — the API (OpenAPI docs).

Every stage is resumable and skips work already done, so re-running is cheap and
Ctrl-C is safe. Extraction is the slow part — roughly an hour on CPU for the full
corpus. For a fast look:

```bash
python run.py --limit 60        # ingest, extract ~60 chunks, link, serve (~15 min CPU)
python run.py --no-extract      # ingest and serve immediately, extract later
```

Requires Python 3.10+ (tested through 3.13; 3.14 also works).

### Going faster: a cloud model or a GPU

All optional — the local CPU path is the default and needs nothing.

- **Gemini — free key, no credit card** (Google AI Studio). ~100× faster than
  local CPU. `./scripts/setup-gemini.sh` walks you through
  <https://aistudio.google.com/apikey> and writes `.env`; then
  `python run.py extract` runs the whole corpus in a couple of minutes. Default
  model `gemini-2.0-flash`.
- **Groq** — `./scripts/setup-groq.sh`, key from <https://console.groq.com>.
  Very fast per call, but the free tier's **output-tokens-per-minute cap is low**
  (~1k), so a full-corpus run gets rate-limited into a crawl — fine for
  incremental uploads, less so for a cold start. Default model
  `qwen/qwen3.8-27b` (Groq rotates its catalogue; set `FKL_GROQ_MODEL` to any
  current chat model if that one is gone —
  `python -c "from groq import Groq; print([m.id for m in Groq().models.list().data])"`).
- Either way: `env.example` has the manual steps, and a real environment
  variable overrides the `.env` file.
- **Local GPU.** GPU offload turns on automatically **iff** the installed
  `llama-cpp-python` has a CUDA/Metal backend. The wheel in `requirements.txt`
  is CPU-only (see [Additional notes](#additional-notes) for why a prebuilt GPU
  wheel is not safe to ship), so a GPU needs a one-time source build:

  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DGGML_AVX512=OFF" \
    pip install --force-reinstall --no-binary :all: llama-cpp-python==0.3.35
  ```

  Needs `cmake` + CUDA toolkit + a host compiler CUDA accepts. Verify with
  `python run.py status` (the backend line notes GPU offload). Force CPU on a
  GPU box with `FKL_N_GPU_LAYERS=0`.

### Uploading your own PDFs

- **UI:** Documents tab → *Add a PDF* → upload. Ingest runs synchronously; then
  extraction and linking run in the background with a progress bar.
- **API:** `curl -F file=@yourfile.pdf http://localhost:8000/api/upload`
- **CLI:** `python run.py ingest path/to/yourfile.pdf && python run.py extract && python run.py link`

Re-uploading identical bytes is a no-op. A new document is linked against
everything already in the store — existing facts are not re-processed.

### Individual commands

```bash
python run.py ingest --dataset all     # parse every starter PDF
python run.py extract [--limit N]      # grounded fact extraction (resumable; --redo to restart)
python run.py link                     # embed facts + classify relationships
python run.py facts | relations | schema
python run.py status                   # what's in the store
python run.py page <document> <page>   # verbatim page text + detected page label
python run.py serve                    # web UI + API only
python -m pytest tests -q
```

### Where to see the four required cases

Open the **Showcase** tab, or `GET /api/showcase`. It selects, by ranked query
over whatever is in the store (nothing hand-picked):

1. a fact **corroborated** across documents, however differently worded;
2. a genuine or likely **contradiction**;
3. an apparent contradiction **explained by context** (period / scope / units);
4. a **documented extraction/reasoning failure** — measured counters plus
   concrete failing chunks, with the narrative and the fixes in
   [DECISIONS.md](DECISIONS.md).

Cases 1–3 show the source quote from *each* document and the system's reasoning,
with a "verify in page context" link that re-checks the stored offsets against
the real page text.

---

## Video demo

**▶ [Demo video (≤3 min)](REPLACE_WITH_LINK)**

Shows a PDF being uploaded and processed, then walks the four required cases in
the Showcase tab — the evidence quotes from each document, the system's
explanation, and the offset verification.

---

## Approach

### Architecture

```
PDF ──ingest──> pages (verbatim text + char offsets) ──chunk──> extraction / retrieval chunks
                                                                      │
                                              per chunk: LLM ──> candidate facts (JSON)
                                                                      │
                                        grounding: quote located char-exact, or fact discarded
                                                                      │
                                    facts + evidence + dynamic fact_types  (SQLite, one file)
                                                                      │
                              embed normalised fact ──> brute-force vector search (cross-document only)
                                                                      │
                        two-step classify (deterministic arithmetic hint + LLM) ──> relationships
                                                                      │
                                      FastAPI  +  single-page web UI  +  /api/showcase
```

One SQLite file holds everything — documents, pages, chunks, facts, evidence,
the `fact_types` registry, vectors, relationships, job state, and the repair log.
No second store to fall out of sync; deleting a document cascades everything
derived from it.

### Important decisions and trade-offs

- **Grounding is the load-bearing invariant.** `pages.text` is exactly what
  PyMuPDF returned; every chunk and quote records offsets into it. Quote matching
  runs in tiers — exact, whitespace-normalised, typography/case-folded, then a
  bounded *reconstructed span* for column-major table text — and every tier
  returns offsets into the original. A quote matching no tier means the fact is
  **dropped**. Trade-off: recall is lower, but a stored fact is always checkable.

- **The schema emerges from the documents.** The local model decodes under a GBNF
  grammar that constrains only the JSON *envelope* — an array of objects with
  arbitrary keys. New attributes become new `fact_types` rows; new payload keys
  are absorbed into a running union. No migrations, no fixed enum.

- **Interchangeable backends** behind one interface: local `llama.cpp` + GGUF
  (default, offline after first run), Gemini or Groq (optional, each active only
  when its API key is present), and a deterministic pattern extractor that is
  the floor the system never falls through. Facts are tagged with which produced
  them. Only the local backend constrains generation with a GBNF grammar; the
  cloud backends rely on the prompt plus `json_guard`'s parse/repair path.

- **Malformed JSON is expected, not exceptional.** Parse → `json-repair` →
  corrective re-prompt → give up, with every outcome written to `repair_log` so
  the repair rate is reported from data, not estimated. Under the GBNF grammar
  the first reply is always syntactically valid JSON, so a parse failure means
  the array truncated at the token ceiling — `json-repair` closes it and keeps
  every fact already emitted, which is far cheaper than a second full
  generation. The re-prompt is the fallback for the rare case repair can't fix.

- **Linking embeds the *normalised fact*, not the sentence** — two documents
  stating the same thing rarely share wording. A brute-force numpy dot product
  over vectors in SQLite beats an ANN index at this scale (exact, ~5 MB matrix)
  and avoids a second store. Only cross-document pairs are considered.

- **Classification is two-step by design.** A deterministic pass settles what
  arithmetic can decide (units match? magnitudes agree? periods differ?) and
  hands that to the LLM as a hint. The rule cannot tell that two differently
  named attributes mean the same thing; the model cannot reliably tell that
  8,142 crore equals 81.42 billion. Each covers the other's blind spot, and the
  rule is the fallback with no model. A `CONTRADICTS` verdict is vetoed to
  `CONTEXT_RECONCILED` when the rule proves the periods or units differ.

- **Table detection is off by default** — measured ~500× slower on dense
  financial reports while producing *worse* output than PyMuPDF's linearised page
  text. Enable with `--tables`.

- **Round-robin chunk ordering.** A partial extraction run interleaves documents,
  so any prefix covers all of them and still demonstrates cross-document links —
  the whole point of the system.

### AI tools used

- **Extraction / relationship classification:** `Qwen2.5-1.5B-Instruct` (Q4_K_M
  GGUF) run locally via `llama-cpp-python`, decoding under a GBNF JSON grammar.
  `Qwen2.5-0.5B` is the low-RAM fallback. Optional cloud drop-ins: Gemini
  (`gemini-2.0-flash`) or Groq (`qwen/qwen3.8-27b`).
- **Embeddings:** `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX runtime, no
  `torch`).
- **JSON recovery:** `json-repair`.
- **Development:** this codebase was built with **Claude Code** (Anthropic) as a
  pair-programming assistant — design discussion, implementation, and the test
  suite. Every measurement and benchmark in [DECISIONS.md](DECISIONS.md) was run
  on the real corpus.

See **[DECISIONS.md](DECISIONS.md)** for the full build diary — the measurements,
the dead ends, and the bugs the real corpus exposed.

---

## Scaling and incrementality

The four extension suggestions in the brief are handled by design, not bolted on:

### Large PDFs without a performance hit

- **Table detection is off by default.** Measured on a 100-page annual report:
  **150.7 s with tables vs 0.3 s without** (~500×), and the detector produces
  *worse* output than PyMuPDF's linearised page text on borderless statistical
  tables. Full ingest of all six PDFs went from **>5 min to 1.7 s**.
- Parsing stores page text once; chunking is O(pages). A cheap numeric-line
  pre-check gates the expensive table detector when it *is* enabled.
- Extraction is per page-sized chunk and **resumable** (`extraction_progress`),
  so a 500-page PDF is never an all-or-nothing operation. `--limit N` bounds a
  run; commits are batched (a kill re-does at most a handful of idempotent
  chunks, never corrupts — WAL + `synchronous=NORMAL`).
- On the model call itself: the fixed part of the extraction prompt is a stable
  prefix so its KV is reused, flash attention is on for any GPU/Metal build, and
  a truncated JSON array is recovered by `json-repair` rather than a second
  generation. The lever that actually collapses wall-clock time on a big job,
  though, is switching the backend to Groq (`./scripts/setup-groq.sh`).

### Many PDFs in one knowledge layer

- One SQLite file holds every document, fact, vector and relationship — no second
  store, no sync problem, deletes cascade.
- Candidate generation pairs facts **only across different documents**;
  same-document pairs (a report repeating its own figures) are excluded before
  the LLM sees them.
- `pending_chunks` interleaves documents **round-robin**, so any partial
  extraction still covers every document and can demonstrate cross-document
  links. Measured: 6/6 documents within the first 60 chunks, vs 1/6 with
  document-order processing.

### A schema that evolves as new fact kinds appear

- The local model decodes under a GBNF grammar that fixes only the JSON
  *envelope* — an array of objects with **arbitrary keys**.
- `fact_types` is a **registry, not a constraint**: a previously unseen
  `attribute` becomes a new row; new payload keys are folded into a running
  `observed_keys` union. **No migration, ever.**
- `payload_json` is schema-free; the promoted columns exist only for indexing.
- Visible in the UI (Schema tab) and at `GET /api/fact-types` — ~170 distinct
  fact types emerged from a partial run over the six starter PDFs (a full run
  yields more).

### New documents incrementally, without rebuilding

- Document identity is `sha256(bytes)`. Re-ingesting identical bytes is a
  **no-op** — nothing reparsed, existing facts and relationships untouched.
- A new document only extracts **its own** chunks (`pending_chunks`), only embeds
  its own facts (`index_facts` skips facts that already have a vector), and only
  classifies **new** candidate pairs (`link_facts` `skip_existing`). The upload
  endpoint scopes the whole extract→link job to the new `document_id`.
- Content-derived fact ids make re-extraction idempotent.
- A changed file hashes differently and becomes a new document, so the old
  version's facts survive and stay comparable against the new ones.
- Tests: `test_pipeline.py::test_adding_a_document_does_not_reprocess_existing_ones`,
  `test_link.py::test_link_facts_skips_a_pair_already_classified`.

Honest limit: the incremental diff is at **document** granularity — a 100-page
PDF changed on one page is reprocessed in full. Page-level hashing would fix it
(see next steps).

---

## Limitations and next steps

- **Extraction quality is bounded by a 1.5B local model.** It still emits some
  non-facts and mis-reads figures on dense chart pages; the grounding check
  catches hallucinated *quotes* but not a plausible wrong number lifted from a
  messy table. Next: ground table facts against `--tables` structured cells, and
  add a numeric-plausibility check against sibling facts of the same key.
- **Incremental ingest diffs at document granularity.** A 100-page PDF changed on
  one page is reprocessed in full. Page-level hashing would fix it.
- **No OCR.** Scanned PDFs are detected and reported (`probably_scanned`) but not
  read.
- **No cross-currency conversion** — deliberately. Inventing an FX rate would
  manufacture false corroborations. Facts in different currencies stay
  numerically incomparable and are left to the LLM to reason about.
- **Candidate generation is O(facts²) per link run.** Fine at this scale
  (thousands of facts); a real ANN index would be needed at 10⁵+.
- **Relationship classification has no human-in-the-loop review** — every
  classified pair is stored with its confidence, but nothing surfaces the
  low-confidence ones for checking.

---

## Additional notes

- **Credentials:** none are required or committed. `.env` is git-ignored and
  loaded automatically if present (`fkl/config.py`); copy `env.example` or run
  `./scripts/setup-groq.sh`. A real environment variable always wins over the
  file. The default path is fully local and offline after the model download.
- **Evaluating without a paid service:** the local backend is the default, so no
  account is needed. `python run.py --limit N` produces a full showcase quickly;
  `sample-output/` contains a captured `/api/showcase` response and
  `/api/failures` response from a full run for reference.
- `requirements.txt` opens with an `--extra-index-url` line serving **prebuilt
  CPU** wheels for `llama-cpp-python`. Without it, pip compiles llama.cpp from
  source (needs `cmake` + a C++ toolchain). With it, install is a plain wheel
  download.
- **Why no prebuilt GPU wheel is shipped:** the public CUDA wheels for
  `llama-cpp-python` are compiled with AVX-512, which SIGILLs on any CPU without
  it (e.g. an Intel 13th-gen laptop part). A GPU build therefore has to come
  from source with `-DGGML_NATIVE=OFF -DGGML_AVX512=OFF` — see
  [Going faster](#going-faster-a-cloud-model-or-a-gpu). Once a CUDA/Metal backend is
  installed, offload is automatic; a normal clone gets the CPU wheel and nothing
  changes.
- `DECISIONS.md` is a running build diary written as the work happened —
  measurements, wrong turns, and the corpus-found bugs — not a reconstruction.
