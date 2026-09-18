"""
test_qa_pipeline.py — Test the full RAG Q&A pipeline against 5 sample questions.

Covers:
  Q1 — Specific merchant lookup   : "How much did I spend at Swiggy?"
  Q2 — Category aggregate         : "What was my total grocery spend in February?"
  Q3 — Cross-month category total : "How much did I spend on subscriptions across all 3 months?"
  Q4 — Income question            : "What was my total income?"
  Q5 — Ambiguous/edge case        : "Are there any transactions I should review?"

For each question we print:
  - The question
  - The chunks the retriever passed to the LLM (so you can verify relevance)
  - The LLM's answer (with its working shown)

This lets you manually verify correctness before the formal evaluator is built.
Run: python scripts/test_qa_pipeline.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

from finsight.categorization.categorizer import categorize_dataframe
from finsight.embeddings.vector_store import build_vector_store
from finsight.ingestion.csv_extractor import load_csv
from finsight.rag.qa_pipeline import ask

SEP  = "═" * 72
SEP2 = "─" * 72

# ── Build pipeline once ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FINSIGHT RAG Q&A — PIPELINE TEST")
print(SEP)

print("\n[1/2] Ingesting and categorizing transactions...")
df = load_csv("data/raw/bank_statement.csv")
df = categorize_dataframe(df)
print(f"      {len(df)} transactions ready\n")

print("[2/2] Building vector store...")
build_vector_store(df)
print()

# ── Ground-truth reference (manually verified from the CSV) ───────────────────
# Use these to check the LLM's answers are correct.
GROUND_TRUTH = {
    "swiggy_total":        420 + 490,            # Jan + Mar = ₹910
    "feb_groceries":       1980 + 2340 + 1750,   # Dmart Feb02 + Bigbasket Feb11 + Dmart Feb21 = ₹6070
    "all_subscriptions":   (199+399+299+130+799)  # Jan
                         + (199+399+299+130+799+119)  # Feb
                         + (199+399+299+130+799+119),  # Mar = ₹4,519
    "total_income":        62000 + 62000 + 62000 + 18500,  # 3x salary + freelance = ₹204,500
}

QUESTIONS = [
    {
        "id":    "Q1",
        "label": "Specific merchant lookup",
        "q":     "How much did I spend at Swiggy in total?",
        "truth": f"₹{GROUND_TRUTH['swiggy_total']:,.2f}",
    },
    {
        "id":    "Q2",
        "label": "Category aggregate — single month",
        "q":     "What was my total grocery spend in February 2026?",
        "truth": f"₹{GROUND_TRUTH['feb_groceries']:,.2f}",
    },
    {
        "id":    "Q3",
        "label": "Category aggregate — all months",
        "q":     "How much did I spend on subscriptions and memberships across all 3 months?",
        "truth": f"₹{GROUND_TRUTH['all_subscriptions']:,.2f}",
    },
    {
        "id":    "Q4",
        "label": "Income question",
        "q":     "What was my total income across January, February, and March?",
        "truth": f"₹{GROUND_TRUTH['total_income']:,.2f}",
    },
    {
        "id":    "Q5",
        "label": "Ambiguous/edge case — should surface Uncategorized rows",
        "q":     "Are there any unrecognized or suspicious transactions I should review?",
        "truth": "Should list the 5 Uncategorized rows (POS PURCHASE, MISC DEBIT, etc.)",
    },
]

# ── Run each question ─────────────────────────────────────────────────────────
for entry in QUESTIONS:
    print(f"\n{SEP}")
    print(f"  {entry['id']} — {entry['label']}")
    print(SEP)
    print(f"  Question : {entry['q']}")
    print(f"  Expected : {entry['truth']}")
    print()

    result = ask(entry["q"])

    if result.error:
        print(f"  ✗ ERROR: {result.error}")
        continue

    # Show what the retriever passed to the LLM
    print(f"  RETRIEVED CHUNKS ({result.n_chunks_used} chunks passed to LLM):")
    print(f"  {SEP2[:50]}")
    for i, hit in enumerate(result.evidence, 1):
        m = hit["metadata"]
        print(
            f"  [{i:02d}] {m['date']}  {m['description'][:38]:<40}  "
            f"₹{abs(m['amount']):>9,.2f}  [{m['category']}]  dist={hit['distance']:.3f}"
        )

    print("\n  LLM ANSWER:")
    print(f"  {SEP2[:50]}")
    # Indent every line of the answer for clean terminal display
    for line in result.answer.splitlines():
        print(f"  {line}")

    print()

print(f"\n{SEP}")
print("  TEST COMPLETE")
print(f"  Model used: {QUESTIONS[0]['id']} → {result.model}")
print(SEP)
print("""
  HOW TO MANUALLY VERIFY:
  1. For Q1-Q4: compare the LLM's '**Answer:**' line against 'Expected'.
  2. For Q5: check the 'Transactions used' section lists Uncategorized rows.
  3. Check 'Transactions used' contains the right rows — if the LLM used
     wrong transactions to reach a correct total, that's still a retrieval bug.
  4. Any 'Confidence: Low / Cannot determine' on Q1-Q4 means retrieval
     is missing relevant transactions — adjust TOP_K_RESULTS in config.py.
""")
