# Fact Knowledge Layer

Extracts grounded facts from PDFs, links every fact to its exact evidence
(page number + verbatim quoted span), and identifies corroborations,
contradictions and context-reconciled relationships across documents.

> **Status: work in progress — stage 1 of 3 complete.**
> This README is a placeholder. The full README (Setup and Run Instructions,
> Video Demo, Approach, Limitations and Next Steps, Additional Notes) ships with
> stage 3. See **[DECISIONS.md](DECISIONS.md)** for the running build diary,
> which is current.

| Stage | Scope | Status |
|---|---|---|
| 1 | PDF parsing with page + character-offset grounding, chunking, SQLite schema, incremental ingest, CLI | ✅ done |
| 2 | LLM backends (llama.cpp / Groq / deterministic), JSON repair, fact extraction, evidence verification, dynamic schema registry | ⏳ next |
| 3 | Embeddings + vector search, relationship classification, FastAPI, web UI, showcase view | ⏳ |

## Quick start (stage 1)

No accounts, no API keys and no background services are required.

```bash
pip install -r requirements.txt

python run.py ingest --dataset all   # parse every starter PDF
python run.py status                 # what's in the store
python run.py page annual-report 41  # verbatim page text + detected page label
python -m pytest tests -q
```

Requires Python 3.10+ (tested through 3.13; 3.14 works).

Ingesting all six starter PDFs — 511 pages, 3,139 chunks — takes **~1.7 seconds**.

## Notes

- `requirements.txt` opens with an `--extra-index-url` line that serves
  **prebuilt** CPU wheels for `llama-cpp-python`. Without it, pip compiles
  llama.cpp from source and needs `cmake` plus a C++ toolchain. With it, install
  is a plain wheel download.
- Embeddings use `fastembed` (ONNX) rather than `sentence-transformers`, to
  avoid a 1–2 GB `torch` dependency. Same models.
- Table detection is **off by default** — it costs ~500× plain text extraction
  on dense financial reports while producing worse output than the plain page
  text. Enable with `--tables`. Full reasoning and measurements in DECISIONS.md.

Built with [Claude Code](https://claude.com/claude-code).
