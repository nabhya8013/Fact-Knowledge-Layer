# DECISIONS.md — build diary

A running log of real decisions, measurements and mistakes, written as they
happened rather than reconstructed afterwards. Numbers in this file are measured
on the actual starter dataset unless explicitly labelled as an estimate.

**Machine used for all timings:** Linux, 20 CPU cores, 15 GiB RAM, no GPU,
Python 3.11.16. Times will differ on a smaller laptop; relative costs will not.

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
30 tests passing
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

_(to be written during stage 2)_

## Stage 3 — Linking, API, UI

_(to be written during stage 3)_
