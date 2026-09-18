<div align="center">

# FinSight AI

**Ask your bank statement anything — get grounded, auditable answers in ₹.**

A retrieval-augmented personal-finance assistant that ingests a bank statement,
categorizes every transaction, and answers natural-language money questions with
the exact transactions it used, so every number is verifiable.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Phase%201-orange)

<!-- TODO: add demo GIF and CI badge once GitHub Actions + Streamlit UI land -->

</div>

---

## The problem

Bank statements are long, messy, and impossible to query. "How much did I spend
on groceries in February?" means scrolling, filtering, and summing by hand — and
most LLM chatbots asked this will *hallucinate* a plausible-but-wrong number.

## The fix

FinSight never lets the LLM guess. Retrieval does the finding; the LLM only adds
up and formats what it's explicitly given. Critically, **aggregate questions
retrieve the *complete* matching set, not a top-K sample** — so totals are exact
rather than silently truncated.

```
                       ┌──────────────────────────────────────────────┐
   bank_statement.csv  │  INGESTION   csv_extractor.load_csv()         │
  ───────────────────► │  tolerant parsing: dates, ₹/$, debit/credit   │
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │  CATEGORIZATION  rule engine (ordered rules)  │
                       │  12 categories + explicit "Uncategorized" flag│
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │  EMBEDDINGS  all-MiniLM-L6-v2 (local, free)   │
                       │  one rich chunk per txn → ChromaDB (persisted)│
                       └───────────────────┬──────────────────────────┘
                                           ▼
   "total grocery     ┌──────────────────────────────────────────────┐
    spend in Feb?" ──►│  QUERY PARSER  → intent, category, month,     │
                      │                  merchant  → metadata filter  │
                      └───────────────────┬──────────────────────────┘
                                           ▼
                      ┌──────────────────────────────────────────────┐
                      │  RETRIEVAL  filtered + COMPLETE (not top-K)   │
                      └───────────────────┬──────────────────────────┘
                                           ▼
                      ┌──────────────────────────────────────────────┐
                      │  GENERATION  GPT-4o-mini, strict system prompt│
                      │  grounded in ₹, shows its working, or says    │
                      │  "I cannot determine"                         │
                      └───────────────────────────────────────────────┘
```

## Highlights

- **Grounded, auditable answers.** A strict system prompt forces the model to
  answer in ₹, compute totals only from the transactions it's given, show which
  rows it used, and say *"I cannot determine"* when data is insufficient.
- **Correct aggregates via metadata filtering.** A pure-parsing query planner
  (`finsight/rag/query_parser.py`) extracts category / month / merchant and
  retrieves **every** matching row — fixing the classic RAG bug where a top-K
  cutoff silently drops transactions and corrupts the total.
- **100% local embeddings.** `all-MiniLM-L6-v2` + ChromaDB run offline — no API
  cost or key needed to index and retrieve.
- **Offline verification.** `scripts/verify_retrieval.py` proves retrieval
  correctness against CSV ground truth with **no LLM call**, so the core is
  testable without an API key.
- **Honest categorization.** Unmatched transactions are flagged `Uncategorized`
  (surfaced for review) rather than buried in a catch-all.
- **Tolerant ingestion.** Handles column-name variants, multiple date formats,
  `(50.00)` negatives, `$`/`₹` symbols, and split debit/credit columns.

## Retrieval accuracy (verified offline)

`python scripts/verify_retrieval.py` compares retrieved totals against the CSV.
All cases return the **complete** matching set:

| Question | Filter applied | Txns | Total | |
|---|---|---:|---:|:--:|
| Total spent at Swiggy | merchant = swiggy | 2 | ₹910 | ✅ |
| Grocery spend in Feb 2026 | category + month | 3 | ₹6,070 | ✅ |
| Subscriptions across all months | category | 14 | ₹4,519 | ✅ |
| Total income (Jan–Mar) | category + months | 4 | ₹2,04,500 | ✅ |
| Transport spending in January | category + month | 3 | ₹3,200 | ✅ |

> The subscriptions case returns **14** rows — a naive top-K=10 retriever would
> have dropped 4 and produced a wrong total.

## Project structure

```
FinSight AI/
├── finsight/
│   ├── config.py                  # all paths, models, categories, constants
│   ├── ingestion/csv_extractor.py # messy CSV → clean DataFrame
│   ├── categorization/categorizer.py  # ordered rule engine
│   ├── embeddings/vector_store.py # chunking, ChromaDB build/query, filters
│   └── rag/
│       ├── query_parser.py        # question → structured retrieval plan
│       └── qa_pipeline.py         # retrieve → prompt → grounded answer
├── scripts/
│   ├── ingest.py                  # CLI: CSV → clean CSV/Parquet + summary
│   ├── verify_retrieval.py        # offline correctness check (no LLM)
│   └── test_*.py                  # runnable smoke demos
├── data/                          # raw / processed / chroma_db (git-ignored)
├── tests/                         # pytest suite
├── pyproject.toml
└── requirements.txt
```

## Getting started

```bash
# 1. Clone and enter the project
git clone <your-repo-url> && cd "FinSight AI"

# 2. Create a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
# source .venv/bin/activate          # macOS/Linux

# 3. Install (editable, so `finsight` is importable everywhere)
pip install -e .                      # or: pip install -r requirements.txt

# 4. (Only needed for answer generation) add your LLM key
cp .env.example .env                  # then put OPENAI_API_KEY=sk-... inside
```

> Retrieval, categorization, and `verify_retrieval.py` work **without** any API
> key. The key is only used by `qa_pipeline.ask()` for final answer generation.

## Usage

**Ingest a statement:**
```bash
python scripts/ingest.py --input data/raw/bank_statement.csv
```

**Verify retrieval (offline, no key):**
```bash
python scripts/verify_retrieval.py
```

**Ask questions (needs `OPENAI_API_KEY`):**
```python
from finsight.ingestion.csv_extractor import load_csv
from finsight.categorization.categorizer import categorize_dataframe
from finsight.rag.qa_pipeline import build_and_ask

df = categorize_dataframe(load_csv("data/raw/bank_statement.csv"))
result = build_and_ask(df, "What was my total grocery spend in February 2026?")

print(result.answer)   # grounded answer with the transactions it used
print(result.scope)    # the metadata filter that was applied
```

## Testing

```bash
pytest                 # unit + retrieval tests
python scripts/verify_retrieval.py   # end-to-end retrieval correctness
```

## Roadmap

- [x] Phase 1 — CSV ingestion, categorization, embeddings, grounded RAG
- [x] Metadata-filtered retrieval for exact aggregates
- [ ] Evaluation harness with retrieval-recall + answer-accuracy metrics
- [ ] Streamlit UI (upload → ask → visualize)
- [ ] PDF bank-statement ingestion (`pdfplumber`)
- [ ] CI (GitHub Actions) + public deployment

## Tech stack

`pandas` · `sentence-transformers` · `ChromaDB` · `OpenAI (GPT-4o-mini)` ·
`Streamlit` (UI) · `pytest` · `scikit-learn` (eval)

## License

MIT
