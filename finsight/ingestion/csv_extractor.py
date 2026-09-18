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

import logging
import re
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

    return clean_transactions(raw_df)


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning chain to an already-read table.

    Shared by the CSV and PDF extractors so both inputs converge on the same
    schema: date, description, amount, transaction_type.
    """
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

    Ordering matters: the alias table maps BOTH "debit" and "credit" to
    "amount", so split columns must be merged first — otherwise the alias
    rename collides two source columns onto the same "amount" name.
    """
    df = df.copy()
    col_lower = {c.lower().strip(): c for c in df.columns}  # lowercase → original name

    # ── 1. Merge split debit/credit BEFORE alias renaming ─────────────────────
    single_amount_aliases = [a for a in CSV_COLUMN_ALIASES["amount"] if a not in ("debit", "credit")]
    has_split         = "debit" in col_lower and "credit" in col_lower
    has_single_amount = any(a in col_lower for a in single_amount_aliases)

    if has_split and not has_single_amount:
        dcol, ccol = col_lower["debit"], col_lower["credit"]
        logger.info("  Detected split debit/credit columns — merging into 'amount'")
        debit  = pd.to_numeric(df[dcol].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
        credit = pd.to_numeric(df[ccol].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0)
        # Convention: debits are money out (negative), credits are money in (positive)
        df = df.drop(columns=[dcol, ccol])
        df["amount"] = credit - debit
        col_lower = {c.lower().strip(): c for c in df.columns}  # refresh after drop

    # ── 2. Alias rename for the remaining columns ───────────────────────────
    alias_lookup = {}
    for standard_name, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lookup[alias.lower()] = standard_name

    rename_map = {}
    for lower_col, original_col in col_lower.items():
        if lower_col not in alias_lookup:
            continue
        # If a single 'amount' column already exists alongside debit/credit,
        # don't also fold debit/credit onto it (would duplicate the name).
        if has_split and has_single_amount and lower_col in ("debit", "credit"):
            continue
        rename_map[original_col] = alias_lookup[lower_col]

    df = df.rename(columns=rename_map)

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
    df = df.dropna(subset=["date"]).copy()
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
    df = df.dropna(subset=["amount"]).copy()
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
