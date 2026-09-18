"""
Preview the exact text that will be stored as embedding chunks.
Run: python scripts/preview_chunks.py
"""
import logging

from finsight.categorization.categorizer import categorize_dataframe
from finsight.ingestion.csv_extractor import load_csv

logging.disable(logging.CRITICAL)   # suppress INFO noise for this preview

df = load_csv("data/raw/bank_statement.csv")
df = categorize_dataframe(df)

def build_chunk(row) -> str:
    """
    The exact format that vector_store.py will embed.
    Packs category + date + merchant + amount + type into one sentence so
    queries about any of those dimensions can retrieve this transaction.
    """
    day_str = row["date"].strftime("%A, %B %d %Y")
    amount  = abs(row["amount"])
    return (
        f"{row['category']} transaction on {day_str}: "
        f"{row['description']} — ₹{amount:.2f}. "
        f"Type: {row['transaction_type']}."
    )

# ── Show 4 representative samples ─────────────────────────────────────────
sample_patterns = [
    ("Groceries",     r"Bigbasket|Dmart"),
    ("Transport",     r"Ola Cabs|Uber India"),
    ("Subscriptions", r"Netflix"),
    ("Income",        r"Infosys|Freelance"),
]

print()
print("=" * 70)
print("  EMBEDDING CHUNK FORMAT — 4 SAMPLE TRANSACTIONS")
print("=" * 70)

for label, pattern in sample_patterns:
    match = df[df["description"].str.contains(pattern, case=False, na=False)]
    if match.empty:
        print(f"\n  [{label}]  (no match for pattern '{pattern}')")
        continue
    row  = match.iloc[0]
    chunk = build_chunk(row)
    print(f"\n  [{label}]")
    print(f"  Raw row  : date={row['date'].date()}  desc='{row['description']}'  amount={row['amount']}  type={row['transaction_type']}")
    print(f"  Chunk    : \"{chunk}\"")

print()
print("=" * 70)
print("  WHAT EACH PART CONTRIBUTES TO RETRIEVAL")
print("=" * 70)
print("""
  'Groceries transaction'   → matches queries about category
  'on Monday, January 05'   → matches queries about specific dates
  'Bigbasket'               → matches queries about specific merchants
  '₹1840.00'                → matches queries about amounts
  'Type: debit'             → matches queries about spending vs income
""")

# ── Show all chunks for a quick full sanity-check ─────────────────────────
print("=" * 70)
print("  ALL 84 CHUNKS (first 15 shown)")
print("=" * 70)
for i, (_, row) in enumerate(df.iterrows()):
    if i >= 15:
        print(f"  ... and {len(df) - 15} more")
        break
    print(f"  {i+1:>2}. {build_chunk(row)}")
print()
