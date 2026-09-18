"""
Test the vector store: build it, then run 3 sample queries and show top-3 results.
Run: python scripts/test_vector_store.py
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

from finsight.categorization.categorizer import categorize_dataframe
from finsight.embeddings.vector_store import build_vector_store, query_vector_store
from finsight.ingestion.csv_extractor import load_csv

# ── 1. Build the store ────────────────────────────────────────────────────────
print("\nStep 1 — Ingesting and categorizing transactions...")
df = load_csv("data/raw/bank_statement.csv")
df = categorize_dataframe(df)
print(f"  {len(df)} transactions ready\n")

print("Step 2 — Building vector store (first run downloads ~80MB model)...")
collection = build_vector_store(df)
print()

# ── 2. Run sample queries and show top-3 results ─────────────────────────────
QUERIES = [
    "grocery purchases",
    "how much did I spend on transport",
    "subscriptions and streaming services",
    "unknown or unrecognized transactions",   # should surface Uncategorized rows
]

SEP = "─" * 72

for query in QUERIES:
    print(SEP)
    print(f"  QUERY: \"{query}\"")
    print(SEP)

    hits = query_vector_store(query, n_results=3)

    for rank, hit in enumerate(hits, 1):
        m = hit["metadata"]
        print(f"\n  #{rank}  [distance: {hit['distance']:.4f}]")
        print(f"       Chunk    : {hit['document']}")
        print(f"       Date     : {m['date']}  |  Category: {m['category']}")
        print(f"       Amount   : ₹{m['amount']:.2f}  |  Type: {m['transaction_type']}")

    print()

print(SEP)
print("  Vector store test complete.")
print(SEP)
print()
print("  NOTE ON DISTANCES:")
print("  Cosine distance scale: 0.0 = identical  |  ~0.3 = highly relevant")
print("                         ~0.6 = loosely related  |  1.0+ = unrelated")
print()
