"""
csv_extractor.py — Ingests a bank statement CSV and returns a clean DataFrame.

Design goals:
  - Be tolerant of real-world CSV messiness (column name variants, date
    format differences, dollar signs, comma-separated amounts, split
    debit/credit columns).
  - Produce one consistent output schema regardless of input format:

        date         | datetime64  — normalized to date-only (no time)
        description  | str         — merchant / narrative, stripped & title-cased
        amount       | float       — positive = credit (money in),
                                     negative = debit  (money out)
        transaction_type | str     — "credit" or "debit"

  - Raise clear errors early so problems are caught at ingest time,
    not silently downstream.
"""

import re
import logging
from pathlib import Path

import pandas as pd
from dateutil import parser as dateutil_parser

# Pull column alias map from central config so we only update it in one place.
from finsight.config import CSV_COLUMN_ALIASES

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Main entry point. Accepts a path to a bank CSV, returns a clean DataFrame.

    Args:
        filepath: Path to the CSV file (string or Path object).

    Returns:
        pd.DataFrame with columns: date, description, amount, transaction_type.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if required columns can't be found after alias resolution.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    logger.info(f"Loading CSV: {filepath.name}")

    raw_df = _read_raw(filepath)
    logger.info(f"  Raw shape: {raw_df.shape} | Columns: {list(raw_df.columns)}")

    df = _normalize_columns(raw_df)
    df = _parse_dates(df)
    df = _parse_amounts(df)
    df = _clean_descriptions(df)
    df = _add_transaction_type(df)
    df = _drop_non_transactions(df)
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"  Clean shape: {df.shape} | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_raw(filepath: Path) -> pd.DataFrame:
    """
    Read the CSV, trying a few common encodings.
    Banks sometimes export UTF-8-BOM or latin-1.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(filepath, encoding=encoding, skip_blank_lines=True)
            # Strip whitespace from column headers — a common issue
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filepath} with any supported encoding.")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map whatever column names the bank used → our standard names
    (date, description, amount).

    Two modes are handled:
      1. Single 'amount' column (positive = credit, negative = debit)
      2. Split 'debit' and 'credit' columns (common in UK/AU bank exports)

    Uses CSV_COLUMN_ALIASES from config.py to match column headers
    case-insensitively.
    """
    # Build a lookup: lowercase_alias → standard_name
    alias_lookup = {}
    for standard_name, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lookup[alias.lower()] = standard_name

    col_lower = {c.lower(): c for c in df.columns}  # lowercase → original name

    rename_map = {}
    for lower_col, original_col in col_lower.items():
        if lower_col in alias_lookup:
            rename_map[original_col] = alias_lookup[lower_col]

    df = df.rename(columns=rename_map)

    # Handle split debit/credit columns → merge into single 'amount'
    has_debit  = "debit"  in df.columns
    has_credit = "credit" in df.columns

    if has_debit and has_credit and "amount" not in df.columns:
        logger.info("  Detected split debit/credit columns — merging into 'amount'")
        df["debit"]  = pd.to_numeric(df["debit"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
        df["credit"] = pd.to_numeric(df["credit"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
        # Convention: debits are money out (negative), credits are money in (positive)
        df["amount"] = df["credit"] - df["debit"]
        df = df.drop(columns=["debit", "credit"])

    # Validate required columns exist
    required = ["date", "description", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in CSV.\n"
            f"Available columns after alias resolution: {list(df.columns)}\n"
            f"Add new aliases in config.py → CSV_COLUMN_ALIASES."
        )

    return df[required]   # keep only what we need


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'date' column into proper datetime64.

    We use dateutil.parser for maximum format tolerance — it handles
    MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD, '01 Jan 2026', etc.
    Rows with unparseable dates are dropped with a warning.
    """
    def _safe_parse(val):
        if pd.isna(val):
            return pd.NaT
        try:
            return dateutil_parser.parse(str(val), dayfirst=False)
        except Exception:
            logger.warning(f"  Could not parse date: '{val}' — row will be dropped")
            return pd.NaT

    df["date"] = df["date"].apply(_safe_parse)
    before = len(df)
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  Dropped {dropped} rows with unparseable dates.")

    # Normalise to date only (strip any time component banks sometimes include)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _parse_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the 'amount' column to float.

    Handles: '$1,234.56', '(50.00)' [parentheses = negative], '1.234,56'
    """
    def _clean_amount(val) -> float | None:
        if pd.isna(val):
            return None
        s = str(val).strip()
        # Parentheses notation  → negative: (50.00) → -50.00
        negative = s.startswith("(") and s.endswith(")")
        # Remove currency symbols, spaces, commas (but keep minus and dot)
        s = re.sub(r"[^\d.\-]", "", s.replace("(", "-").replace(")", ""))
        try:
            result = float(s)
            return -abs(result) if negative and result > 0 else result
        except ValueError:
            logger.warning(f"  Could not parse amount: '{val}' — will be NaN")
            return None

    df["amount"] = df["amount"].apply(_clean_amount)
    before = len(df)
    df = df.dropna(subset=["amount"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  Dropped {dropped} rows with unparseable amounts.")
    df["amount"] = df["amount"].astype(float)
    return df


def _clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise merchant/description text:
      - Strip leading/trailing whitespace
      - Collapse multiple internal spaces
      - Title-case for consistent display
    """
    df["description"] = (
        df["description"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )
    return df


def _add_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 'transaction_type' from the sign of 'amount'.
    Downstream modules (e.g. the RAG pipeline) can filter on this
    without having to re-check amount signs.
    """
    df["transaction_type"] = df["amount"].apply(
        lambda x: "credit" if x > 0 else "debit"
    )
    return df


def _drop_non_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that aren't real transactions:
      - Zero-amount rows (e.g. 'Opening Balance' placeholders)
      - Rows where description is blank or NaN
    """
    before = len(df)
    df = df[df["amount"] != 0.0]
    df = df[df["description"].str.strip() != ""]
    df = df.dropna(subset=["description"])
    dropped = before - len(df)
    if dropped:
        logger.info(f"  Dropped {dropped} zero-amount or blank-description rows.")
    return df
