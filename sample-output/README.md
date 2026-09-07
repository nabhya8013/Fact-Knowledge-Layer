# Sample output

Captured from a real run so the system can be evaluated without running the
local model. **No paid service is involved** — this is the default local backend
(`Qwen2.5-1.5B-Instruct` GGUF) — the sample is here purely to save reviewer time.

## Run configuration

| | |
|---|---|
| Corpus | the six starter PDFs (511 pages, 3,139 chunks) |
| Extraction backend | `Qwen2.5-1.5B-Instruct-Q4_K_M` GGUF, local, GBNF-constrained |
| Extraction scope | 124 of 674 numeric chunks (a partial run — round-robin ordered, so all 6 documents are covered evenly). A full run produces more facts; the four cases and the behaviour shown here are representative. |
| Embeddings | `BAAI/bge-small-en-v1.5` via `fastembed` |
| Similarity threshold | 0.82 |
| Result | 331 facts, 331 evidence rows, 168 fact types, 129 cross-document relationships |

## Files

| File | What it is |
|---|---|
| `showcase.json` | `GET /api/showcase` — the four required cases, selected by ranked query |
| `showcase.txt` | the same four cases, human-readable |
| `failures.json` | `GET /api/failures` — measured extraction/reasoning failure surface (Case 4) |
| `relationships.json` | `GET /api/relationships?limit=200` — every classified pair with both facts, evidence and reasoning |
| `fact-types.json` | `GET /api/fact-types` — the dynamic schema registry (168 types that emerged from the documents) |
| `documents.json` | `GET /api/documents` — per-document stats |
| `health.json` | `GET /api/health` |
| `cli-status.txt` | `python run.py status` |
| `cli-relations-contradicts.txt` | `python run.py relations --type CONTRADICTS` and `--type CORROBORATES` |
| `cli-facts-sample.txt` | `python run.py facts --numeric-only --limit 20` |

## The four cases at a glance

1. **Corroboration** — RBI Annual Report: *"real gross domestic product (GDP) growth moderated to 6.5 per cent in 2024-25"* vs IMF Article IV: *"India's real GDP grew by 6.5 percent in FY2024/25."* Same value, same period, two institutions, different wording.
2. **Contradiction** — Economic Survey 2024-25: *"India's real GDP is estimated to grow by 6.4 per cent in FY25"* vs IMF Article IV: *"India's real GDP grew by 6.5 percent in FY2024/25."* Same period, same metric, 6.4 % vs 6.5 %.
3. **Context-reconciled** — RBI: IMF global growth *"3.3 per cent in 2024"* vs Economic Survey: IMF *"3.2 per cent and 3.3 per cent for 2024 and 2025"*. The apparent gap is a period/scope difference.
4. **Failure** — see `failures.json` / `showcase.txt`: 124 chunks attempted, 49 produced no grounded fact (the model wrote text not in the document — those facts were discarded), the JSON re-prompt/repair path fired on ~48 % of chunks (none abandoned), and `reconstructed_span` is the noisiest grounding tier. A specific failure we found and fixed (a chart-caption mis-extraction and a period-mismatch mislabelled as a contradiction) is written up in `../DECISIONS.md`.
