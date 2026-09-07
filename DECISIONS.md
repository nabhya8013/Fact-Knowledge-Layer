# DECISIONS.md — build diary

A running log of real decisions, measurements and mistakes, written as they
happened rather than reconstructed afterwards. Numbers in this file are measured
on the actual starter dataset unless explicitly labelled as an estimate.

**Machine used for stage-1 timings:** Linux, 20 CPU cores, 15 GiB RAM, no GPU,
Python 3.11.16. Stage 2/3 extraction timings are given for both the CPU path and
a single RTX 4060 (used for development speed; the shipped default is still CPU).
Times will differ on a smaller laptop; relative costs will not.

---

## Stage 0 — Environment reconnaissance (before writing any code)

### The `llama-cpp-python` install problem, and why it is not a problem

The brief requires a cold `git clone && pip install -r requirements.txt &&
python run.py` to work with no background services. The obvious risk was the
local LLM runtime, so I checked it first rather than discovering it at the end.

```
$ pip download llama-cpp-python --no-deps --only-binary=:all:
ERROR: Could not find a version that satisfies the requirement llama-cpp-python
       (from versions: none)
```

**`llama-cpp-python` publishes no binary wheels to PyPI at all.** A plain
`pip install` always compiles llama.cpp from source, which needs `cmake` and a
C++ toolchain. The dev machine has `gcc` but neither `g++` nor `cmake`, so the
canonical install would have failed outright on the very machine building it —
and would fail for any recruiter without a full build environment.

The fix was to use the project's own prebuilt CPU wheel index:

```
$ pip download llama-cpp-python --no-deps --only-binary=:all: \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
Saved llama_cpp_python-0.3.35-py3-none-manylinux2014_x86_64.whl
```

Note the wheel tag: **`py3-none`**, not `cp311-cp311`. `llama-cpp-python` binds
to `libllama.so` through `ctypes` rather than the CPython C-API, so one wheel
covers every Python 3.x. That turned a feared blocker into the most portable
part of the stack. Verified availability:

| Platform | Wheel |
|---|---|
| Linux x86_64 | `0.3.35` `py3-none-manylinux2014_x86_64` |
| macOS arm64 (Apple Silicon) | `0.3.35` `py3-none-macosx_11_0_arm64` |
| Windows amd64 | `0.3.35` `py3-none-win_amd64` |
| macOS x86_64 (Intel) | only `0.2.90` `cp312` — lags badly |

Consequence: `requirements.txt` opens with an `--extra-index-url` line. That one
line is the difference between "installs in 40 seconds" and "fails unless you
have a C++ compiler". Intel macs are the one documented soft spot.

### Dropping `sentence-transformers` for `fastembed`

The brief suggested `sentence-transformers`, which depends on `torch` (~1–2 GB
download). `fastembed` serves the same models (`all-MiniLM-L6-v2`,
`bge-small-en-v1.5`) through `onnxruntime` instead, with no torch. For a
CPU-only, cold-clone install that is a large win in install size and first-run
time, with the same embeddings. Documented as a deliberate deviation.

### Python version

The dev machine defaults to Python 3.14, which is newer than most wheels target.
Checked directly: `pymupdf` ships `cp310-abi3`, `chromadb` ships `cp39-abi3`,
`fastembed`/`sentence-transformers` are pure Python, `torch` now has `cp314`
wheels, and `llama-cpp-python`'s prebuilt wheel is `py3-none`. So 3.14 actually
works. `run.py` targets 3.10+, warns above 3.13, and refuses below 3.10 with an
actionable message instead of a stack trace.

### The starter documents

All six PDFs are text-bearing — no OCR needed. Confirmed by extracting a middle
page from each and checking character counts. The system still detects and
*reports* a likely-scanned document (`is_probably_scanned`) rather than silently
producing nothing, because arbitrary future PDFs will not be so cooperative.

---

## Stage 1 — Ingestion

### Chunking: two roles, not one

The first instinct is a single chunking strategy. That is wrong here, because
extraction and retrieval want opposite things:

* **Extraction** wants *non-overlapping* units. Overlap means the same sentence
  is seen twice, which produces duplicate facts that then have to be
  de-duplicated — worse, two near-identical facts from overlapping windows look
  like a corroborating pair to the linker, manufacturing fake corroboration.
* **Retrieval** wants *overlapping* windows, so a fact that straddles a boundary
  is still findable by similarity search.

So the store holds both, distinguished by `chunks.role`. Measured on the starter
set: 722 extraction chunks and 2,417 retrieval chunks over 511 pages.

Extraction granularity is **one page**, split further only if the page exceeds
~900 tokens. A page of these documents averages ~500–800 tokens, comfortably
inside a 4096-token context, and one call per page is roughly 3× fewer LLM calls
than one call per sliding window. Splitting escalates through paragraph →
sentence → line → hard character boundaries, and every level preserves absolute
character offsets into the page text.

**Offsets are the load-bearing invariant.** `pages.text` stores exactly what
PyMuPDF returned — never stripped, normalised or re-wrapped — and every chunk
records `char_start`/`char_end` into that string. A property test asserts
`page_text[char_start:char_end] == chunk_text` for every prose chunk in a real
document. Without that guarantee, stage 2 cannot honestly claim a fact is
grounded to an exact span.

### Table extraction: measured, then turned off by default

This was the biggest surprise of stage 1. Initial implementation ran PyMuPDF's
`page.find_tables()` on every page. The full 6-PDF ingest did not finish within
five minutes. Profiling one document:

| Document | tables ON | tables OFF |
|---|---|---|
| Delhivery Annual Report FY24 (100pp) | **150.7 s** | **0.3 s** |
| Delhivery Prospectus 2022 (100pp) | 22.7 s | 0.2 s |

A **~500× penalty** on the dense document. `cProfile` put the time in
`table.py:char_in_bbox` (547k calls on a 27-page deck alone) and `extract_cells`.

Two fixes went in, then a third decision:

1. **A cheap pre-check** (`_looks_tabular`) that counts predominantly-numeric
   lines before paying for the real detector. Generic — no document is
   special-cased. It helps on prose-heavy PDFs but **barely helped here**: 74 of
   100 pages of a financial report look tabular, so the annual report still took
   135.9 s. Honest result: the heuristic is right in principle and nearly
   useless on this corpus.
2. **Building markdown from the already-extracted rows** instead of calling
   `Table.to_markdown()`, which internally re-runs cell extraction. Roughly
   halves the remaining cost.
3. **Defaulting table detection OFF**, opt-in via `--tables` /
   `FKL_ENABLE_TABLES=1`.

Point 3 deserves justification beyond speed, because "it was slow" is not on its
own a good reason to drop table structure. Checking the actual output, PyMuPDF's
detector on these borderless statistical tables either collapses the whole table
into a single cell (`strategy="lines"`) or shatters column boundaries
(`strategy="text"` produced `|T|able 4. India: Centr|al Gove|rnment|`).
Meanwhile the *plain page text* already linearises the same table usefully:

```
Table 4. India: Central Government Operations, 2021/22–2026/27 1/
2021/22
2022/23
...
Revenue
9.4  9.0  9.3  9.2  9.0  8.9
```

Row-major, label followed by its series — which a small LLM handles fine. So the
default keeps the faithful cheap representation and skips the expensive mangled
one. A shape gate (≥2 rows × ≥2 cols) still rejects the single-cell collapse
when `--tables` *is* enabled.

Result: full ingest of all six PDFs went from **>5 minutes to 1.7 seconds**.

### Page numbers: three separate bugs, all found by measuring coverage

The brief demands evidence cite a page number. The number a reader cares about
is the one *printed on the page*, which in curated excerpts is not the physical
page index — PDF page 50 of the prospectus is printed page 221. PyMuPDF reported
no embedded page labels (`doc.get_page_labels() == []`), so this has to be
recovered from the text.

First implementation: accept a line that is *only* a page number, within 4 lines
of either page edge, and require a neighbouring page to carry the adjacent
number (so stray figures like "Table 4" or a year cannot qualify). Measured
coverage was the tell:

| Document | v1 coverage |
|---|---|
| Delhivery Annual Report | **0%** |
| IMF Article IV | **12%** |

Both were real bugs, and coverage measurement is what exposed them — the code
looked correct in isolation.

**Bug 1 — alternating header styles.** The IMF report alternates between
`INDIA / 38 / INTERNATIONAL MONETARY FUND` (number on its own line, detected)
and `INTERNATIONAL MONETARY FUND 37` (number at the end of a header line,
missed). Catching only one style halves coverage, and because the algorithm
requires *neighbour agreement*, half coverage does not degrade to 50% — it
collapses to near zero, since a detected page's neighbours are exactly the
undetected ones. Fixed by also accepting a short (≤60 char) line that begins or
ends with a number.

**Bug 2 — scan window too narrow.** The Delhivery annual report puts four lines
of running furniture ("Delhivery Limited / Statutory Reports / Corporate
Overview / Financial Statements") at the page foot, pushing the page number
outside a 4-line window. Widened to 8 lines.

**Bug 3 — 2-up spreads.** After fixing 1 and 2, the annual report's labels came
out `81, 83, 85, 87…`, incrementing by two. That looked like a detector bug but
was not: the report is typeset **A3 landscape (1191×842), two printed pages per
PDF page**, so physical page 40 genuinely carries printed pages 80 *and* 81, and
each page yields two valid candidates. Silently picking one half would mis-cite
evidence. Now, when the surviving candidates form a consecutive run, the label is
recorded as a range (`"80-81"`).

Coverage after all three fixes:

| Document | coverage | sample |
|---|---|---|
| Delhivery Prospectus 2022 | 98% | `116, 212, 213, 214` (jump is real — curated excerpt) |
| Delhivery Annual Report FY24 | 100% | `80-81, 82-83, 84-85` (2-up spreads) |
| Delhivery Q4 FY24 deck | 67% | slides often have no page number — expected |
| Economic Survey 2024-25 | 99% | `83, 84, 85, 86` |
| RBI Annual Report 2024-25 | 100% | `36, 37, 38, 39` |
| IMF Article IV 2025 | 94% | `36, 37, 38, 39` |

The neighbour-agreement rule is deliberately conservative: it would rather
return `None` than invent a page number. `pdf_page_index` is always stored and
always correct, so a missing printed label degrades citation quality but never
breaks grounding.

### Incremental ingestion

Document identity is `sha256(file bytes)`. Re-ingesting identical bytes is a
no-op — verified by a test asserting the store's row counts are byte-identical
before and after, and that the second call is faster because nothing is parsed.
A changed file hashes differently and becomes a new document, so the old
version's facts survive and remain comparable against the new ones. Bumping
`PARSER_VERSION` marks previously-parsed documents stale and re-parses them
automatically, without touching documents already current.

Honest limitation: the diff is at **document** granularity. A 100-page PDF with
one changed page is reprocessed in full. Page-level hashing would fix it and is
listed in next steps.

### Measured result for stage 1

```
6 PDFs | 511 pages | 722 extraction + 2,417 retrieval chunks | 1.7 s total
30 tests passing at this point (140 across all three stages)
```

### A finding that shapes stage 2

The `min_numeric_density` filter was meant to skip chunks with no quantitative
content and so cut LLM calls. Measured on the real corpus:

| threshold | extraction chunks | tokens |
|---|---|---|
| 0.00 | 722 | 424,449 |
| 0.02 | 674 | 401,108 |
| 0.05 | 624 | 369,822 |
| 0.10 | 554 | 322,939 |

It is **nearly a no-op** — these documents are numeric almost everywhere, which
in hindsight is obvious for financial filings and macroeconomic reports. So the
filter cannot be the answer to the runtime budget. ~674 chunks at a realistic
1.5–3 s per CPU inference is 17–34 minutes, well past the "few minutes" target.
Stage 2 has to solve this a different way (measuring real tokens/sec first, then
choosing among: smaller model, batching several chunks per call, a bounded
default demo scope with full runs behind a flag). Recorded here so the decision
is made against measurements rather than guesses.

---

## Stage 2 — Fact extraction

### Model choice: Qwen2.5-1.5B-Instruct Q4_K_M

Tried three local models against the grounding check (does the quote the model
returns actually occur in the text it was shown?):

| Model | Size | Quotes that ground | Speed on a dense chunk |
|---|---|---|---|
| Qwen2.5-0.5B-Q4 | ~0.4 GB | **0%** | ~6 s |
| Qwen2.5-1.5B-Q4 | ~1.0 GB | ~80% | ~25 s |
| Phi-3-mini-4k-Q4 | ~2.3 GB | ~80% | ~20 s |

The 0.5B model is unusable here for a specific reason: it parrots the few-shot
example instead of reading the document, so *every* quote it produced was about
"Northwind Freight Ltd" and grounded to nothing. That 0% is actually the
strongest early evidence that the grounding check does real work — it caught
100% of those hallucinations.

Between 1.5B and Phi-3-mini the deciding factors were size (half the download and
half the RAM floor, which is felt immediately on a cold CPU clone) and that
Qwen follows "return only JSON, no prose" markedly better. Phi-3 tends to
prepend a sentence of explanation that then has to be stripped. `0.5B` stays
wired in as `fallback_model_file`, used automatically when free RAM is below
`min_free_ram_gb` — a bad small model still beats an `OutOfMemory` crash.

### The grammar constrains the envelope, not the keys

The local backend decodes under a GBNF grammar (`JSON_ARRAY_GRAMMAR` in
`local_llama.py`). The grammar describes *an array of objects with arbitrary
string keys* and nothing more. Pinning the key set would be the single biggest
thing that could defeat the brief's "schema emerges from the documents"
requirement, so structure is enforced and vocabulary is left completely free.

Measured cost of the grammar: 6.15 s/chunk with it, 6.74 s without — it is free,
and it removes an entire class of "model emitted almost-JSON" failure before it
can happen.

### JSON recovery, and reporting how often it fires

Even under a grammar a small quantised model truncates output at the token limit
mid-string. So `json_guard.py` applies, in order: parse as-is (after stripping
markdown fences, which the model adds despite the grammar — it emits them as
string *content*); a corrective re-prompt that shows the model its own output
and the parser error; then `json-repair` for mechanical damage (trailing commas,
unterminated strings); then give up and count the chunk as failed.

Every outcome is written to `repair_log`, so the repair rate in the failure
report is measured, not estimated. **A bug found here and fixed:** a clean
first-try parse is deliberately *not* logged (it would dominate the table and
cost a write per chunk), but `failure_report` was computing
`repair_rate = repaired / len(repair_log)` — which is `repaired / repaired`,
structurally always `1.0`. It now divides by the number of chunks actually
attempted (from `extraction_progress`). This is itself a small reasoning failure
worth recording: a metric that looked plausible on the dashboard was meaningless.

### Grounding: four tiers, and why the fourth exists

`grounding.py` matches a model's quote back to the source in tiers, each
returning offsets into the *original* page text:

1. `exact` — a genuine substring.
2. `normalised_whitespace` — forgives PyMuPDF's mid-sentence hard wraps, which a
   model asked to "copy verbatim" renders as spaces. Without this tier almost
   every true quote is rejected.
3. `normalised_chars` — forgives curly quotes, en/em dashes, case.
4. `reconstructed_span` — for column-major table text. PyMuPDF emits a table's
   row label, its value and the column's unit header in three non-contiguous
   places, so the model reasonably writes "Total equity 59,798.47 INR million" —
   every piece of which is on the page, nowhere together. This tier finds the
   smallest window of *real* source text containing the quote's anchor tokens
   and stores that window. The model's reconstruction is never stored as if it
   were a quote.

Tier 4 measured on the six most numeric chunks in the corpus: grounding went
from 26% to 89%, with hallucinated figures still rejected (verified by test).
Guard rails on tier 4 are covered under Case 4 below — an early version was too
permissive.

A quote matching no tier is ungrounded and its fact is **discarded, not stored**.
That is the load-bearing rule of the whole system.

### Two bugs the real corpus exposed in value normalisation

- `parse_number`'s separator-grouped regex alternative used `*` where it needed
  `+`, so it matched just the leading 1–3 digits and, because alternation is
  ordered, won before the plain-number branch — silently turning "8142" into 814
  and "2024" into 202. Every 4+ digit unseparated number in the corpus would
  have been corrupted.
- `parse_number` also read digits out of anything: the CIN
  "U63090DL2011PLC221234" became 63090, an address became 24, "Plot 5, Sector
  44…" became 5. Those fabricated magnitudes would have been compared against
  real figures and invented contradictions out of postcodes. `is_quantity` now
  gates it — a token mixing letters and digits is an identifier, and a real
  quantity leaves almost nothing behind once its number, scale word and currency
  are stripped.

### Parallelism: processes, and one worker on a GPU

A `Llama` instance is not safe to call concurrently and llama.cpp releases the
GIL during inference, so threads buy nothing. Separate processes each holding
their own model do scale, bounded by free RAM (~1.8 GB resident per worker) as
well as cores. Fork is unsafe once a model or GPU context is loaded, so the pool
uses `spawn`.

`n_threads` is capped at 6, not `os.cpu_count()`: measured, 20 threads ran ~4×
slower than 6 on this model — memory-bandwidth thrashing. "Use every core" was an
active pessimisation.

On a GPU the pool drops to **one** worker: several processes on one device just
serialise on it and add VRAM pressure. Measured: GPU 10.4 s/chunk vs CPU
22.5 s/chunk on identical work. A full-corpus extraction is ~117 min on one GPU
worker, or ~63 min on CPU at 4 workers.

### Resumability

`extraction_progress` holds one row per attempted chunk. `pending_chunks`
selects those not yet `done`, so a re-run continues where an interrupt stopped —
verified in practice by killing a run mid-corpus and confirming it picked up the
remainder exactly once. Fact ids are content-derived, so re-extracting a chunk
cannot duplicate facts. `--redo` clears progress and starts over deliberately.

---

## Stage 3 — Linking, API, UI

### Embed the normalised fact, not the sentence

Two documents stating the same fact rarely share wording:

    "revenue from contract with customers was Rs 8,142 crore in FY24"
    "Revenue        81,420      (INR million)"

Embedding the raw text puts those far apart. What gets embedded is
`canonical_text(payload)` — `"Delhivery - revenue: 8142 INR crore [FY24]"` —
built from the fact's fields, so the two land next to each other. Embedding facts
rather than passages is the whole point.

`fastembed`/ONNX rather than `sentence-transformers`, to avoid a 1–2 GB `torch`
download for the same `bge-small-en-v1.5` model.

### A numpy index in SQLite, not Chroma

The brief allowed either. At this scale — a few thousand facts × 384 dims is a
~5 MB matrix — a brute-force dot product is single-digit milliseconds, exact
rather than approximate, and genuinely faster than building an ANN index.
Storing the vectors in the existing SQLite file means there is no second store to
fall out of sync, no separate persistence directory, and deleting a document
still cascades its vectors away for free. `chromadb` was therefore removed from
`requirements.txt` — it was pinned but never imported, and it contradicted the
"keep the install small" rationale that drove the `fastembed` choice.

### Similarity threshold: 0.62 → 0.82

The first threshold was a guess. Calibrated against real BGE embeddings of corpus
facts, BGE turned out to have a high similarity floor: genuinely unrelated facts
(a revenue figure vs a GDP-growth figure) still score ~0.63, so 0.62 would have
admitted essentially every pair. The same fact expressed in crore vs million
scores ~0.97. 0.82 sits in the gap.

### Two-step classification

A deterministic pass runs first and settles what arithmetic can decide: do the
canonical keys match, are the units comparable, do the magnitudes agree, do the
periods differ. That verdict is handed to the LLM as a *hint*, not used directly
— because arithmetic cannot tell that "revenue from operations" and "revenue
from contract with customers" are the same property, while the model cannot
reliably tell that 8,142 crore equals 81.42 billion. Each covers the other's
blind spot, and the deterministic verdict is also the fallback when no model is
loaded.

Relationships are stored as first-class rows (`relationships` table) — both fact
ids, type, machine-readable `reason_tag`, natural-language `explanation`,
confidence, similarity — queryable by type, by reason, or by either fact. A
graph rendering alone was explicitly called out as insufficient.

### Two concurrency bugs found by running commands during a live extraction

- `connect()` executed the full schema script on every open, taking a write lock
  each time. During a long extraction that locked out every read-only CLI command
  and would have locked out the web UI. It now runs the schema only when the
  stored `schema_version` differs, so WAL's many-readers-one-writer actually
  applies.
- Linking held a single transaction open across ten classified chunks — roughly
  two minutes at CPU inference speed, far past the busy timeout — so a concurrent
  reader hit "database is locked". Commits are now per pair; with WAL and
  `synchronous=NORMAL` that costs nothing next to inference.

### Chunk ordering so a partial run still demonstrates links

`pending_chunks` originally ordered by document. A partial run — and on CPU a
full run is ~an hour, so partial runs are the common case — then produced facts
from only the first document or two, and therefore **zero** cross-document
relationships, which are the entire point. Chunks are now interleaved
round-robin: the first chunk of every document, then the second of every
document, and so on. Measured: 6 of 6 documents covered within the first 60
chunks, against 1 before.

### API: a connection per request

FastAPI runs sync endpoints in a threadpool and SQLite connections are not
shareable across threads, so a module-level connection would be a latent
corruption bug, not an optimisation. Every request opens and closes its own
connection; opening one is microseconds and WAL means readers never block on the
background writer. Long operations (upload → extract → link) are dispatched to a
worker thread and tracked in a `jobs` table the UI polls.

---

## Case 4 — an extraction-and-reasoning failure we found, and how we handled it

**What happened.** On a partial extraction run, the showcase's single
`CONTRADICTS` example was wrong twice over.

The Q4 FY24 earnings deck has a chart captioned, in PyMuPDF's linearised output:

    Revenue from services* (INR million)
    72,236  FY23  70,536  FY22  81,415  FY24  27,748

The model was asked for a `revenue` fact and quoted the whole caption, reporting
the value as **27,748** — which is not Delhivery's FY24 revenue from services
(that is 81,415; 27,748 appears to be a quarter figure). This grounded through
the `reconstructed_span` tier because every numeral in the quote *was* present
on the page, within the (then 400-character) window.

The linker then paired that fact with a `revenue` fact from the annual report
and the model classified it `CONTRADICTS` with reason "different_value" — but the
two facts were for **different periods** (FY24 vs FY21). A period mismatch is
`CONTEXT_RECONCILED` by definition; it cannot be a contradiction.

**How we handled it.**

1. `find_reconstructed_span` guard rails tightened: it now refuses a quote
   carrying more than four distinct numerals (that is a chart region, not one
   fact), requires 75% word coverage (was 60%), and caps the window at 240
   characters (was 400). The caption above no longer grounds, so the bad fact is
   never stored — a missing fact is better than a wrong number.
2. `classify_pair` gained a deterministic veto: if the model returns
   `CONTRADICTS` but the `time_scope` or the normalised unit of the two facts
   differ, the verdict is downgraded to `CONTEXT_RECONCILED`, keeping the model's
   explanation but correcting the label. A contradiction requires everything
   except the value to match, and the rule can prove it does not.
3. The `json_repair_rate` metric bug described under Stage 2 — a reasoning
   failure in our own reporting rather than in the pipeline — was fixed at the
   same time.

Both fixes have regression tests (`tests/test_extract.py`,
`tests/test_link.py`), and the failure report now surfaces concrete
`grounding_failures` and `reconstructed_span_examples` in the UI so a reviewer
can see the noisiest tier at work rather than just its count.

**How we would improve it further.**

- Ground table facts against `--tables` structured cells rather than linearised
  page text. The linearised form is cheap and faithful for prose, but for a
  borderline chart it discards exactly the row/column structure that says which
  number is "the" value.
- A numeric-plausibility check: when a new fact shares a `canonical_key` with
  facts already in the store, flag it low-confidence if its magnitude is a large
  outlier against them. The corpus itself becomes a sanity check.
- The 1.5B model is the real ceiling on extraction quality. The Groq backend
  (`llama-3.3-70b-versatile`, enabled with `GROQ_API_KEY`) already exists as the
  higher-quality path when a reviewer wants it.
