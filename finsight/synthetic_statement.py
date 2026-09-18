"""
synthetic_statement.py — Seeded generator for realistic bank statements.

Why this exists
───────────────
Real personal bank statements are never published (privacy), and the "bank"
datasets on public mirrors (UCI bank marketing, fraud/AML tables) are
customer or campaign records — they contain no transaction ledger at all.
Rather than demo against a toy file of one fixed size, this module generates
reproducible ledgers in exactly the schema the ingestion pipeline expects:

    Date         | YYYY-MM-DD
    Description  | merchant / narrative text
    Amount       | signed float, positive = credit, negative = debit

Reproducibility matters: the same (seed, start month, months) always yields
byte-identical output, so tests and evaluation runs are deterministic.

Realism levers
──────────────
  - Fixed-day recurring events (salary, rent, SIP, subscriptions, bills)
  - Stochastic merchant visits with per-merchant frequency and amount ranges
  - Occasional extra credits (freelance work, refunds from friends)
  - Description "messiness" (ALL CAPS exports, stray double spaces) so the
    cleaning chain in csv_extractor is exercised, not bypassed
"""

from __future__ import annotations

import calendar
import random
from datetime import date

import pandas as pd

# ── Recurring events: (day_of_month, description, signed amount) ─────────────
# Day is clamped to the month length (e.g. day 30 in February).
RECURRING: list[tuple[int, str, float]] = [
    (1, "Salary Credit - Tech Corp Pvt Ltd", 68000.00),
    (3, "Rent Payment - Property Owner", -18000.00),
    (5, "Mutual Fund Sip Transfer", -5000.00),
    (7, "Spotify Premium", -119.00),
    (8, "Act Fibernet Broadband", -999.00),
    (10, "Bescom Electricity Bill", -1450.00),   # jittered ±15% below
    (12, "Netflix Subscription", -649.00),
    (15, "Amazon Prime Membership", -299.00),
    (20, "Google One Storage", -130.00),
]

# ── Stochastic spending: (description, min, max, min/month, max/month) ───────
SPENDING: list[tuple[str, float, float, int, int]] = [
    ("Bigbasket - Grocery Delivery", 450, 2600, 2, 3),
    ("Dmart - Groceries", 380, 1900, 1, 2),
    ("Zepto Quick Delivery", 180, 900, 1, 3),
    ("Local Kirana Store", 90, 600, 1, 3),
    ("Swiggy Order", 160, 680, 3, 6),
    ("Zomato Order", 180, 720, 1, 4),
    ("Box8 Meal Delivery", 220, 480, 0, 2),
    ("Third Wave Coffee", 140, 420, 0, 3),
    ("Uber Trip", 90, 460, 3, 7),
    ("Indian Oil Petrol", 400, 2200, 1, 3),
    ("Metro Rail Card Topup", 100, 500, 0, 2),
    ("Bmtc Bus Pass", 110, 220, 0, 1),
    ("Amazon.In - Online Shopping", 299, 4200, 1, 3),
    ("Flipkart Order", 249, 3200, 0, 2),
    ("Myntra Fashion Order", 499, 2600, 0, 2),
    ("Adidas India - Online Shopping", 800, 3400, 0, 1),
    ("Pvr Cinemas Ticket", 260, 900, 0, 2),
    ("Bookmyshow Event", 400, 1800, 0, 1),
    ("Steam Game Purchase", 250, 1200, 0, 1),
    ("Apollo Pharmacy", 120, 850, 0, 2),
    ("Dr Sharma Family Clinic", 300, 1500, 0, 1),
    ("ATM Withdrawal - Cash", 500, 2000, 0, 1),
    ("Bank Charge - SMS Alerts", 25, 60, 0, 1),
]

# ── Occasional extra credits: (description, min, max, min/month, max/month) ──
CREDITS: list[tuple[str, float, float, int, int]] = [
    ("Freelance Payment - Client Project", 4000, 15000, 0, 1),
    ("UPI Credit - Refund From Friend", 200, 1500, 0, 1),
]


def _style(rng: random.Random, description: str) -> str:
    """Mimic export quirks so the cleaning chain has real work to do."""
    roll = rng.random()
    if roll < 0.15:
        return description.upper()
    if roll < 0.25:
        return description.lower()
    if roll < 0.32:
        return description.replace(" ", "  ", 1)
    return description


def _iter_months(start: date, months: int):
    year, month = start.year, start.month
    for _ in range(months):
        yield year, month
        month += 1
        if month == 13:
            month, year = 1, year + 1


def generate_statement(
    months: int = 6,
    seed: int = 42,
    start: date | None = None,
) -> pd.DataFrame:
    """
    Build a synthetic transaction ledger.

    Args:
        months: Number of calendar months to generate.
        seed:   RNG seed — identical (seed, start, months) ⇒ identical output.
        start:  First day of the first month. Defaults to the current month
                minus (months - 1), i.e. "the last N months".

    Returns:
        pd.DataFrame with columns Date (str YYYY-MM-DD), Description, Amount,
        sorted by date — the same shape a raw bank CSV export would have.
    """
    if months < 1:
        raise ValueError("months must be >= 1")

    rng = random.Random(seed)
    if start is None:
        today = date.today().replace(day=1)
        month_index = today.year * 12 + (today.month - 1) - (months - 1)
        start = date(month_index // 12, month_index % 12 + 1, 1)

    rows: list[tuple[str, str, float]] = []

    for year, month in _iter_months(start, months):
        days_in = calendar.monthrange(year, month)[1]

        for day, description, amount in RECURRING:
            if "Electricity" in description:
                amount = round(amount * rng.uniform(0.85, 1.15), 2)
            rows.append((date(year, month, min(day, days_in)).isoformat(),
                         _style(rng, description), amount))

        for table, sign in ((SPENDING, -1.0), (CREDITS, 1.0)):
            for description, lo, hi, n_min, n_max in table:
                for _ in range(rng.randint(n_min, n_max)):
                    amount = round(rng.uniform(lo, hi), 2) * sign
                    rows.append((date(year, month, rng.randint(1, days_in)).isoformat(),
                                 _style(rng, description), amount))

    rows.sort(key=lambda r: r[0])
    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])
