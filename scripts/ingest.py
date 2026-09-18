"""
scripts/ingest.py — CLI entry point for the ingestion pipeline.

Usage:
    python scripts/ingest.py                          # uses default CSV in data/raw/
    python scripts/ingest.py --input data/raw/my_statement.csv
    python scripts/ingest.py --input data/raw/my_statement.csv --output data/processed/clean.csv

What it does:
    1. Loads and normalises the raw CSV via csv_extractor.load_csv()
    2. Prints a summary table and key stats to the terminal
    3. Saves the clean DataFrame to data/processed/ as both CSV and Parquet
       (Parquet preserves dtypes like datetime — useful when other modules load it)
"""

import argparse
import logging
import sys
from pathlib import Path

from finsight.config import RAW_DIR, PROCESSED_DIR, CURRENCY_SYMBOL
from finsight.ingestion.csv_extractor import load_csv

# ── Logging setup ─────────────────────────────────────────────────────────────
# INFO level gives us the per-step messages we added in csv_extractor.py
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinSight AI — Ingest a bank statement CSV and produce a clean table."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=RAW_DIR / "bank_statement.csv",
        help="Path to the raw bank statement CSV (default: data/raw/bank_statement.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path for the cleaned CSV output (default: data/processed/<input_stem>_clean.csv)",
    )
    return parser.parse_args()


def print_summary(df) -> None:
    """Print a human-readable summary of what was extracted."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("  EXTRACTION SUMMARY")
    print(sep)
    print(f"  Total transactions : {len(df)}")
    print(f"  Date range         : {df['date'].min().date()}  →  {df['date'].max().date()}")

    credits = df[df["transaction_type"] == "credit"]
    debits  = df[df["transaction_type"] == "debit"]

    print(f"  Credits (money in) : {len(credits):>4}  |  Total: {CURRENCY_SYMBOL}{credits['amount'].sum():>10,.2f}")
    print(f"  Debits  (money out): {len(debits):>4}  |  Total: {CURRENCY_SYMBOL}{abs(debits['amount'].sum()):>10,.2f}")
    print(f"  Net cash flow      :             {CURRENCY_SYMBOL}{df['amount'].sum():>10,.2f}")

    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")["amount"]
        .agg(
            transactions="count",
            net_flow="sum",
            total_spent=lambda x: abs(x[x < 0].sum()),
            total_income=lambda x: x[x > 0].sum(),
        )
        .reset_index()
    )
    for _, row in monthly.iterrows():
        print(
            f"    {row['month']}  |  {int(row['transactions']):>2} txns  "
            f"|  income {CURRENCY_SYMBOL}{row['total_income']:>8,.2f}  "
            f"|  spent {CURRENCY_SYMBOL}{row['total_spent']:>8,.2f}  "
            f"|  net {CURRENCY_SYMBOL}{row['net_flow']:>8,.2f}"
        )

    # Sample: first 10 rows
    print(f"\n  First 10 transactions:")
    print(f"  {'DATE':<12} {'DESCRIPTION':<45} {'AMOUNT':>10}  TYPE")
    print(f"  {'─'*12} {'─'*45} {'─'*10}  {'─'*6}")
    for _, row in df.head(10).iterrows():
        amt_str = f"{CURRENCY_SYMBOL}{row['amount']:>9,.2f}" if row["amount"] >= 0 else f"-{CURRENCY_SYMBOL}{abs(row['amount']):>8,.2f}"
        desc = row["description"][:44]  # truncate long merchant names
        print(f"  {str(row['date'].date()):<12} {desc:<45} {amt_str}  {row['transaction_type']}")

    print(sep)


def main() -> None:
    args = parse_args()

    # ── 1. Extract ─────────────────────────────────────────────────────────────
    logger.info(f"Starting ingestion: {args.input}")
    try:
        df = load_csv(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # ── 2. Print summary to terminal ───────────────────────────────────────────
    print_summary(df)

    # ── 3. Save processed output ───────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Default output path derives from input filename
    if args.output is None:
        out_csv     = PROCESSED_DIR / f"{args.input.stem}_clean.csv"
    else:
        out_csv     = args.output

    out_parquet = out_csv.with_suffix(".parquet")

    # Drop the helper 'month' column we added for the summary
    save_df = df.drop(columns=["month"], errors="ignore")

    save_df.to_csv(out_csv, index=False)
    save_df.to_parquet(out_parquet, index=False)

    logger.info(f"Saved clean CSV     → {out_csv}")
    logger.info(f"Saved clean Parquet → {out_parquet}")
    print(f"\n  ✓ Done. Clean data saved to:\n    {out_csv}\n    {out_parquet}\n")


if __name__ == "__main__":
    main()
