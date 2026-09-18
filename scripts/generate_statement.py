"""
scripts/generate_statement.py — CLI for the synthetic statement generator.

Usage:
    python scripts/generate_statement.py                          # 6 months, seed 42
    python scripts/generate_statement.py --months 24 --seed 7
    python scripts/generate_statement.py --months 12 --output data/raw/year_statement.csv
    python scripts/generate_statement.py --start 2025-04          # pin the first month

Why: real personal bank statements are never public, so reproducible
synthetic ledgers are how the pipeline gets demoed and stress-tested at
any size. Same (seed, start, months) ⇒ byte-identical CSV.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from finsight.config import RAW_DIR
from finsight.synthetic_statement import generate_statement

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinSight AI — generate a reproducible synthetic bank statement CSV."
    )
    parser.add_argument("--months", "-m", type=int, default=6,
                        help="Number of calendar months to generate (default: 6)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--start", type=str, default=None,
                        help="First month as YYYY-MM (default: ends at the current month)")
    parser.add_argument("--output", "-o", type=Path, default=RAW_DIR / "generated_statement.csv",
                        help="Output CSV path (default: data/raw/generated_statement.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = None
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m").date()

    df = generate_statement(months=args.months, seed=args.seed, start=start)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    credits = df[df["Amount"] > 0]
    debits = df[df["Amount"] < 0]
    logger.info(f"Generated {len(df)} transactions "
                f"({df['Date'].min()} → {df['Date'].max()})")
    logger.info(f"  credits: {len(credits)}  total +{credits['Amount'].sum():,.2f}")
    logger.info(f"  debits : {len(debits)}  total {debits['Amount'].sum():,.2f}")
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
