"""
Quick smoke-test for categorizer.py.
Run: python scripts/test_categorizer.py
Shows categorization on the full dataset + edge cases.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

from finsight.ingestion.csv_extractor import load_csv
from finsight.categorization.categorizer import (
    categorize_dataframe,
    get_category_summary,
)
from finsight.config import PROCESSED_DIR

# ── Load clean data ────────────────────────────────────────────────────────
df = load_csv("data/raw/bank_statement.csv")
df_cat = categorize_dataframe(df)

# Save categorized output
PROCESSED_DIR.mkdir(exist_ok=True)
out = PROCESSED_DIR / "bank_statement_categorized.csv"
df_cat.to_csv(out, index=False)
print(f"\n  Saved → {out}")

# ── Show edge-case transactions you specifically asked about ──────────────
edge_cases = [
    "Bigbasket",
    "Dmart",
    "Ola Cabs",
    "Uber India",
    "Netflix",
    "Mutual Fund Sip",
    "Swiggy",
    "Zomato",
    "Cult.Fit",
    "Apollo Pharmacy",
    "Dr Sharma",
    "Amazon.In",
    "Bescom",
    "Google One",
    "Indian Oil",
    "Atm Withdrawal",
]

print("\n" + "─" * 72)
print("  EDGE-CASE SPOT CHECK")
print("─" * 72)
print(f"  {'DESCRIPTION':<45} {'CATEGORY':<20}")
print(f"  {'─'*45} {'─'*20}")

for fragment in edge_cases:
    matches = df_cat[df_cat["description"].str.contains(fragment, case=False, na=False)]
    if matches.empty:
        # Show what the categorizer returns for a synthetic edge case
        from finsight.categorization.categorizer import categorize_transaction
        cat = categorize_transaction(fragment)
        print(f"  {fragment:<45} {cat:<20}  (synthetic — not in data)")
    else:
        row = matches.iloc[0]
        desc = row["description"][:44]
        print(f"  {desc:<45} {row['category']:<20}")

# Add a couple of genuinely ambiguous synthetic descriptions
print()
synthetic_ambiguous = [
    "POS PURCHASE #4821",
    "MISC DEBIT 0042",
    "PAYMENT - REF 9920",
    "DEBIT MEMO 1183",
    "TRSF DR 77412",
]
print(f"  {'─'*45} {'─'*20}")
print(f"  Synthetic ambiguous cases:")
print(f"  {'─'*45} {'─'*20}")
from finsight.categorization.categorizer import categorize_transaction
for desc in synthetic_ambiguous:
    cat = categorize_transaction(desc)
    print(f"  {desc:<45} {cat:<20}")

# ── Category summary table ────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  FULL CATEGORY SUMMARY  (all 84 transactions)")
print("─" * 72)
summary = get_category_summary(df_cat)
print(f"  {'CATEGORY':<22} {'TOTAL':>10}  {'TXNS':>5}  {'% SPEND':>8}")
print(f"  {'─'*22} {'─'*10}  {'─'*5}  {'─'*8}")
for _, row in summary.iterrows():
    total_str = f"₹{row['total_amount']:>9,.2f}"
    print(f"  {row['category']:<22} {total_str}  {int(row['txn_count']):>5}  {row['pct_of_spending']:>8}")
print("─" * 72)
