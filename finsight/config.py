"""
config.py — Central configuration for FinSight AI.

All paths, model names, and tunable constants live here.
Import this wherever you need settings; never hardcode these values
in individual modules.

    from finsight.config import RAW_DIR, TOP_K_RESULTS
"""

from pathlib import Path

# ── Project root (the directory that CONTAINS the finsight package) ───────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SAMPLES_DIR    = DATA_DIR / "samples"

# ── Vector store ─────────────────────────────────────────────────────────────
CHROMA_DIR     = DATA_DIR / "chroma_db"   # ChromaDB persists here

# ── Embedding model (local, no API cost) ─────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # ~80 MB, fast, good quality

# ── LLM for answer generation ────────────────────────────────────────────────
# Supported: "gemini" | "openai" | "anthropic"
LLM_PROVIDER   = "gemini"
LLM_MODEL      = "gemini-3.6-flash"      # free tier, fast, strong vision + text
# Set your key in the environment, e.g.:  export GEMINI_API_KEY="AIza..."
# or create a .env file (never commit it to git). Each provider reads its own
# env var: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.

# ── Categorization ────────────────────────────────────────────────────────────
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Shopping",
    "Transport",
    "Utilities",
    "Rent/Housing",
    "Healthcare",
    "Entertainment",
    "Subscriptions",
    "Savings/Transfer",
    "Income",
    "Other",
    "Uncategorized",   # rule engine found no match — needs human review
]

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 10   # how many transactions to retrieve per question

# ── Currency ──────────────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "₹"   # used in chunk text, summaries, and LLM prompts
CURRENCY_NAME   = "INR" # used in LLM prompt instructions (see qa_pipeline.py)

# ── CSV ingestion ─────────────────────────────────────────────────────────────
# Column name aliases: maps common bank CSV headers → our standard names.
# Add more aliases here if you try a real bank export and columns differ.
CSV_COLUMN_ALIASES = {
    "date":        ["date", "transaction date", "trans date", "posted date", "value date"],
    "description": ["description", "merchant", "payee", "details", "narrative", "memo"],
    "amount":      ["amount", "debit", "credit", "transaction amount", "value"],
}
