"""
categorizer.py — Assigns a spending category to each transaction.

Architecture
────────────
Rule-based matching only (Phase 1). The engine does a case-insensitive
substring search of each transaction description against an ordered list
of keyword rules.

Two important distinctions:
  "Other"         → matched, but genuinely miscellaneous (ATM withdrawals,
                    donations, parking, etc.). We *know* what it is; it just
                    doesn't fit a named category.
  "Uncategorized" → the rule engine found NO match at all. This is a signal
                    that the rules need extending, not a silent catch-all.
                    These rows should be reviewed before trusting any totals.

Why ordered rules (list of tuples) instead of a dict?
  Order matters. "Groceries" rules are checked before "Food & Dining" so
  that "Wholesome Market - Groceries" hits the right bucket. A dict gives
  no ordering guarantees.

Public API
──────────
  categorize_transaction(description: str) -> str
  categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame
"""

import re
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Rule table ────────────────────────────────────────────────────────────────
# Format: (category_name, [keyword_fragments, ...])
# Rules are checked top-to-bottom; FIRST match wins.
# Keywords are matched as case-insensitive substrings of the description.
# Add new keywords here as you encounter new merchants — this is the only
# file you need to edit to improve categorization accuracy.

RULES: list[tuple[str, list[str]]] = [

    # ── Income ────────────────────────────────────────────────────────────────
    # Check income first so salary/freelance credits never fall into anything else
    ("Income", [
        "salary credit",
        "salary",
        "payroll",
        "freelance payment",
        "client payment",
        "neft credit",
        "imps credit",
        "upi credit",
        "interest earned",
        "dividend",
        "refund",
        "cashback",
    ]),

    # ── Savings / Transfers ───────────────────────────────────────────────────
    ("Savings/Transfer", [
        "mutual fund sip",
        "sip transfer",
        "transfer to savings",
        "savings transfer",
        "transfer to",           # generic account-to-account transfer
        "account transfer",
        "internal transfer",
        "wire transfer",
    ]),

    # ── Rent / Housing ────────────────────────────────────────────────────────
    ("Rent/Housing", [
        "rent payment",
        "rent",
        "apartment",
        "lease payment",
        "home loan emi",
        "housing society",
        "maintenance charges",
        "society maintenance",
    ]),

    # ── Groceries ─────────────────────────────────────────────────────────────
    # Before "Food & Dining" — Indian grocery platforms share food-related words
    ("Groceries", [
        "bigbasket",
        "dmart",
        "d-mart",
        "grofers",
        "blinkit",
        "zepto",
        "jiomart",
        "more megastore",
        "reliance fresh",
        "reliance smart",
        "spencer",
        "nature's basket",
        "lulu hypermarket",
        "star bazaar",
        "hypercity",
        "grocery",
        "supermarket",
        "kirana",
        "vegetables",
        "fruits and vegetables",
    ]),

    # ── Food & Dining ─────────────────────────────────────────────────────────
    ("Food & Dining", [
        "zomato",
        "swiggy",
        "box8",
        "faasos",
        "freshmenu",
        "rebel foods",
        "eatfit",
        "mcdonald",
        "kfc india",
        "burger king india",
        "domino",
        "pizza hut india",
        "subway india",
        "barbeque nation",
        "mainland china",
        "paradise biryani",
        "barbeque",
        "starbucks",
        "chai point",
        "cafe coffee day",
        "ccd",
        "third wave coffee",
        "blue tokai",
        "restaurant",
        "dining",
        "bistro",
        "dhaba",
        "hotel",                # many Indian eateries use "hotel" in their name
        "tiffin",
        "mess",
        "food delivery",
        "meal delivery",
        "valentine",            # "Valentine Dinner" style entries
    ]),

    # ── Transport ─────────────────────────────────────────────────────────────
    ("Transport", [
        "ola cabs",
        "ola ",                 # "OLA " with space to avoid "COLA" etc.
        "uber india",
        "uber ",
        "rapido",
        "namma yatri",
        "yulu",
        "bounce",
        "bluesmart",
        "metro rail",
        "bmtc",                 # Bengaluru Metropolitan Transport Corporation
        "best bus",             # Mumbai
        "dtc bus",              # Delhi
        "ksrtc",
        "bus pass",
        "fastag",
        "toll plaza",
        "parking",
        "indian oil",
        "iocl",
        "hp petrol",
        "bharat petroleum",
        "bpcl",
        "shell india",
        "petrol pump",
        "fuel station",
        "irctc",                # train bookings
        "redbus",
        "abhibus",
        "makemytrip flight",
        "indigo airlines",
        "air india",
        "spicejet",
    ]),

    # ── Utilities ─────────────────────────────────────────────────────────────
    ("Utilities", [
        "bescom",               # Bengaluru electricity
        "msedcl",               # Maharashtra electricity
        "tata power",
        "adani electricity",
        "electricity bill",
        "electric bill",
        "water bill",
        "gas bill",
        "piped gas",
        "mgl",                  # Mahanagar Gas
        "igl",                  # Indraprastha Gas
        "airtel broadband",
        "airtel fiber",
        "jio fiber",
        "act fibernet",
        "hathway",
        "bsnl broadband",
        "broadband",
        "internet bill",
        "postpaid bill",        # catches "JIO POSTPAID BILL", "AIRTEL POSTPAID"
    ]),

    # ── Subscriptions ─────────────────────────────────────────────────────────
    # After utilities so Jio/Airtel postpaid doesn't land here
    ("Subscriptions", [
        "netflix",
        "spotify",
        "hotstar",
        "disney+ hotstar",
        "amazon prime",
        "zee5",
        "sony liv",
        "sonyliv",
        "voot",
        "mxplayer",
        "youtube premium",
        "google one",
        "google play",
        "microsoft 365",
        "adobe",
        "dropbox",
        "membership",           # catches "CULT.FIT MEMBERSHIP" etc.
        "subscription",
        "cult.fit",
        "curefit",
    ]),

    # ── Healthcare ────────────────────────────────────────────────────────────
    ("Healthcare", [
        "apollo pharmacy",
        "medplus",
        "netmeds",
        "1mg",
        "pharmeasy",
        "tata 1mg",
        "pharmacy",
        "hospital",
        "fortis",
        "manipal",
        "narayana health",
        "columbia asia",
        "max healthcare",
        "consultation",
        "clinic",
        "dr. ",                 # "Dr. Sharma" style — dot+space is specific enough
        "family clinic",
        "sharma",               # named doctor entries — extend as needed
        "diagnostic",
        "thyrocare",
        "lal pathlabs",
        "dental",
        "dentist",
        "medical",
        "health insurance",
        "star health",
        "niva bupa",
        "hdfc ergo health",
    ]),

    # ── Entertainment ─────────────────────────────────────────────────────────
    ("Entertainment", [
        "pvr",                  # PVR Cinemas
        "inox",                 # INOX cinemas
        "cinepolis",
        "bookmyshow",
        "district by zomato",   # live events
        "paytm insider",
        "lbb",                  # Little Black Book events
        "concert",
        "bowling",
        "escape room",
        "steam ",
        "playstation",
        "xbox",
        "nintendo",
        "games",
        "kindle",
        "audible",
        "museum",
        "zoo",
        "aquarium",
        "spa",
        "salon",
        "nykaa salon",
    ]),

    # ── Shopping ─────────────────────────────────────────────────────────────
    # Broad — check after all specific ones above
    ("Shopping", [
        "amazon.in",            # .in to avoid matching "Amazon Prime" subscription
        "flipkart",
        "myntra",
        "ajio",
        "meesho",
        "snapdeal",
        "nykaa",
        "tata cliq",
        "reliance digital",
        "croma",
        "vijay sales",
        "ikea india",
        "home centre",
        "lifestyle stores",
        "max fashion",
        "pantaloons",
        "westside",
        "h&m india",
        "zara india",
        "adidas india",
        "nike india",
        "puma india",
        "online shopping",      # fallback for generic descriptions
        "fashion shopping",
    ]),

    # ── Other ─────────────────────────────────────────────────────────────────
    # Known miscellaneous — NOT a silent catch-all for unrecognized merchants
    ("Other", [
        "atm withdrawal",
        "cash withdrawal",
        "donation",
        "charity",
        "ngo",
        "temple donation",
        "bank charge",
        "bank fee",
        "overdraft",
        "late payment fee",
        "interest charge",
        "cheque bounce",
        "neft charges",
    ]),

    # NOTE: "Uncategorized" is NOT in this list.
    # It is the fallback returned by categorize_transaction() when nothing matches.
]


# ── Public API ────────────────────────────────────────────────────────────────

def categorize_transaction(description: str) -> str:
    """
    Match a single transaction description against the rule table.

    Returns the category name string, or "Uncategorized" if no rule matched.
    This is intentionally NOT "Other" — callers can tell the difference.

    Args:
        description: merchant name / narrative text (will be lowercased internally)

    Returns:
        Category string, e.g. "Groceries", "Transport", "Uncategorized"
    """
    desc_lower = description.lower()

    for category, keywords in RULES:
        for kw in keywords:
            if kw.lower() in desc_lower:
                return category

    # Nothing matched → flag it, don't bury it
    logger.debug(f"No rule matched: '{description}' → Uncategorized")
    return "Uncategorized"


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorize_transaction() to every row of the clean transactions DataFrame.

    Expects the schema produced by csv_extractor.load_csv():
        date, description, amount, transaction_type

    Returns the same DataFrame with a new 'category' column appended.
    The original DataFrame is not mutated (we work on a copy).

    Args:
        df: Clean transactions DataFrame from the ingestion module.

    Returns:
        pd.DataFrame with an added 'category' column.
    """
    result = df.copy()
    result["category"] = result["description"].apply(categorize_transaction)

    # Log a summary so problems are visible immediately
    total = len(result)
    uncategorized = (result["category"] == "Uncategorized").sum()
    by_cat = result["category"].value_counts()

    logger.info(f"Categorization complete: {total} transactions")
    if uncategorized > 0:
        logger.warning(
            f"  {uncategorized} transactions are Uncategorized — "
            f"review these and add keywords to RULES in categorizer.py"
        )
        for desc in result[result["category"] == "Uncategorized"]["description"]:
            logger.warning(f"    → '{desc}'")
    else:
        logger.info("  All transactions matched a rule ✓")

    logger.info("  Category distribution:")
    for cat, count in by_cat.items():
        logger.info(f"    {cat:<20} {count:>3} txns")

    return result


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: summarise spending by category.

    Takes a categorized DataFrame (output of categorize_dataframe) and
    returns a summary with total spent, transaction count, and % of total
    spending per category.

    Income and Savings/Transfer are excluded from the spending base so that
    percentages reflect actual expenditure only — e.g. rent showing 45% means
    45% of what you *spent*, not 45% of all money that moved through the account.
    Both categories still appear as rows in the summary table for completeness.

    Args:
        df: Categorized DataFrame (must have 'category' and 'amount' columns).

    Returns:
        pd.DataFrame with columns: category, total_amount, txn_count, pct_of_spending
    """
    # These categories are excluded from the spending denominator
    NON_SPENDING = {"Income", "Savings/Transfer"}

    # Build the summary across ALL categories (so Income and Savings/Transfer
    # still show up as rows — just with "—" in the pct column)
    summary = (
        df.groupby("category")
        .agg(
            total_amount=("amount", "sum"),
            txn_count=("amount", "count"),
        )
        .reset_index()
        .sort_values("total_amount")  # most negative (biggest spend) first
    )

    # Spending base = only debit categories, excluding non-spending ones
    total_spent = abs(df[~df["category"].isin(NON_SPENDING)]["amount"].sum())

    def _pct(row) -> str:
        if row["category"] in NON_SPENDING:
            return "—"
        if row["total_amount"] >= 0:
            return "—"
        if total_spent == 0:
            return "—"
        return f"{abs(row['total_amount']) / total_spent * 100:.1f}%"

    summary["pct_of_spending"] = summary.apply(_pct, axis=1)

    return summary
