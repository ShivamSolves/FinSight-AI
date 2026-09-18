"""
query_parser.py — Turns a natural-language question into a structured retrieval plan.

Why this exists
───────────────
Pure semantic top-K retrieval silently breaks aggregate questions. Ask
"total subscription spend across all months" and a top-10 cutoff drops rows,
so the LLM sums an incomplete list and returns a confident-but-wrong number.

The fix is to read the question for *hard filters* — a category, a month, a
specific merchant — and an *intent* (does the user want a complete set to add
up, or just a few relevant examples?). qa_pipeline uses this plan to query
ChromaDB with a metadata `where` clause and, when completeness matters, to
fetch ALL matching rows instead of a top-K slice.

This module is pure and offline: no LLM, no vector store, no I/O. That makes it
trivially unit-testable and lets us verify retrieval correctness without an API
key.

Public API
──────────
  parse_query(question) -> ParsedQuery
  build_where(parsed)   -> dict | None   (ChromaDB metadata filter)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight.config import CATEGORIES

# ── Month vocabulary ──────────────────────────────────────────────────────────
# name/abbrev (lowercase) → month number. Abbreviations are matched on a word
# boundary so "mar" doesn't fire inside "market".
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# month number → full name (for human-readable scope descriptions)
_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
    12: "December",
}

# ── Category triggers ─────────────────────────────────────────────────────────
# Synonyms/colloquialisms → canonical category name (must exist in config.CATEGORIES).
# Order within a category doesn't matter; the first category with ANY matching
# trigger wins, checked in the order listed here (most specific first).
_CATEGORY_TRIGGERS: list[tuple[str, list[str]]] = [
    ("Rent/Housing", ["rent", "housing", "lease", "emi", "maintenance"]),
    ("Groceries", ["grocery", "groceries", "supermarket", "kirana", "vegetables",
                   "provision", "provisions"]),
    ("Subscriptions", ["subscription", "subscriptions", "streaming", "membership",
                       "ott", "netflix", "spotify", "hotstar", "prime video"]),
    ("Transport", ["transport", "transportation", "travel", "cab", "cabs", "taxi",
                   "uber", "ola", "metro", "bus", "train", "flight", "fuel",
                   "petrol", "diesel", "commute", "fastag", "toll"]),
    ("Utilities", ["utility", "utilities", "electricity", "bill", "bills",
                   "broadband", "internet", "water bill", "gas bill", "postpaid",
                   "bescom", "power bill"]),
    ("Healthcare", ["health", "healthcare", "medical", "medicine", "pharmacy",
                    "hospital", "doctor", "clinic", "dental", "diagnostic"]),
    ("Entertainment", ["entertainment", "movie", "movies", "cinema", "theatre",
                       "theater", "concert", "event", "games", "gaming", "spa",
                       "salon"]),
    ("Food & Dining", ["food", "dining", "restaurant", "restaurants", "eat",
                       "eating", "dinner", "lunch", "breakfast", "cafe", "coffee",
                       "swiggy", "zomato", "order", "takeaway", "snacks"]),
    ("Shopping", ["shopping", "shop", "purchase", "purchases", "online order",
                  "amazon", "flipkart", "myntra", "clothes", "apparel", "electronics"]),
    ("Savings/Transfer", ["savings", "saving", "transfer", "sip", "mutual fund",
                          "investment", "investments"]),
    ("Income", ["income", "salary", "payroll", "earnings", "credited", "freelance",
                "revenue"]),
]

# ── Merchant brands ───────────────────────────────────────────────────────────
# Specific merchant names. When one of these appears we filter by merchant
# (substring on description) rather than by category — it's more precise.
# "swiggy" beats "Food & Dining" for "how much did I spend at Swiggy".
_MERCHANTS = [
    "swiggy", "zomato", "bigbasket", "dmart", "d-mart", "zepto", "blinkit",
    "jiomart", "grofers", "netflix", "spotify", "hotstar", "amazon prime",
    "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa", "croma",
    "uber", "ola", "rapido", "irctc", "redbus", "makemytrip", "indigo",
    "spicejet", "air india", "starbucks", "chaos point", "chai point",
    "cafe coffee day", "ccd", "mcdonald", "kfc", "domino", "pizza hut",
    "subway", "barbeque nation", "pvr", "inox", "bookmyshow", "cult.fit",
    "curefit", "apollo pharmacy", "medplus", "netmeds", "1mg", "pharmeasy",
    "bescom", "tata power", "airtel", "jio", "act fibernet", "hathway",
    "ikea", "westside", "pantaloons", "zara", "adidas", "nike", "puma",
]

# ── Aggregate / completeness signals ──────────────────────────────────────────
_AGGREGATE_WORDS = [
    "total", "sum", "how much", "how many", "count", "average", "avg",
    "overall", "spent", "spend", "spending", "all", "across", "combined",
    "altogether", "in total", "grand total",
]

# Phrases that explicitly mean "no month restriction" — don't infer one.
_ALL_MONTHS_PHRASES = ["all month", "all months", "every month", "each month",
                       "across all", "all 3 month", "all three month",
                       "whole year", "entire year", "year to date", "ytd"]


@dataclass
class ParsedQuery:
    """Structured retrieval plan derived from a question."""
    raw:            str
    intent:         str                       # "aggregate" | "lookup"
    categories:     list[str] = field(default_factory=list)
    merchant:       str | None = None
    months:         list[str] = field(default_factory=list)   # "YYYY-MM" (year known)
    month_nums:     list[int] = field(default_factory=list)   # 1-12 (year unknown)
    wants_all:      bool = False              # fetch every matching row, not top-K

    @property
    def has_filter(self) -> bool:
        return bool(self.categories or self.merchant or self.months or self.month_nums)

    def describe_scope(self) -> str:
        """Human-readable summary of what was filtered — shown to the LLM."""
        parts = []
        if self.merchant:
            parts.append(f"merchant contains '{self.merchant}'")
        if self.categories:
            parts.append("category = " + ", ".join(self.categories))
        if self.months:
            parts.append("month = " + ", ".join(self.months))
        elif self.month_nums:
            names = [_MONTH_NAMES[m] for m in self.month_nums]
            parts.append("month = " + ", ".join(names))
        if not parts:
            return "no metadata filter (ranked by semantic relevance)"
        return "; ".join(parts)


def _find_months(q: str) -> tuple[list[str], list[int]]:
    """Return (year-qualified 'YYYY-MM' list, bare month-number list)."""
    lower = q.lower()
    if any(p in lower for p in _ALL_MONTHS_PHRASES):
        return [], []   # user explicitly wants all months → no month filter

    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    year = years[0] if years else None

    nums: list[int] = []
    for name, num in _MONTHS.items():
        # word-boundary match so "mar" doesn't hit "market", "may" doesn't hit "maybe"
        if re.search(rf"\b{re.escape(name)}\b", lower):
            if num not in nums:
                nums.append(num)

    if not nums:
        return [], []
    if year:
        return [f"{year}-{n:02d}" for n in sorted(nums)], []
    return [], sorted(nums)


def _find_merchant(q: str) -> str | None:
    lower = q.lower()
    # longest brand first so "amazon prime" beats "amazon"
    for brand in sorted(_MERCHANTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand)}\b", lower):
            return brand
    return None


def _find_categories(q: str) -> list[str]:
    lower = q.lower()
    found: list[str] = []
    for category, triggers in _CATEGORY_TRIGGERS:
        if category not in CATEGORIES:
            continue
        for t in triggers:
            if re.search(rf"\b{re.escape(t)}\b", lower):
                if category not in found:
                    found.append(category)
                break
    return found


def _is_aggregate(q: str) -> bool:
    lower = q.lower()
    return any(w in lower for w in _AGGREGATE_WORDS)


def parse_query(question: str) -> ParsedQuery:
    """
    Parse a question into a ParsedQuery retrieval plan.

    Precedence: a specific merchant beats a category (more precise). Months are
    resolved to year-qualified strings when a year is present, otherwise to bare
    month numbers. `wants_all` is set whenever a hard filter is present or the
    question is aggregate — in both cases the caller needs the COMPLETE matching
    set, not a top-K sample.
    """
    merchant   = _find_merchant(question)
    categories = [] if merchant else _find_categories(question)
    months, month_nums = _find_months(question)
    aggregate  = _is_aggregate(question)

    wants_all = aggregate or bool(merchant or categories or months or month_nums)

    return ParsedQuery(
        raw=question,
        intent="aggregate" if aggregate else "lookup",
        categories=categories,
        merchant=merchant,
        months=months,
        month_nums=month_nums,
        wants_all=wants_all,
    )


def build_where(parsed: ParsedQuery) -> dict | None:
    """
    Build a ChromaDB metadata `where` clause from a ParsedQuery.

    Merchant is NOT encoded here (ChromaDB has no substring operator); it's
    applied as a post-filter in query_vector_store. Returns None when there's
    nothing to filter on.
    """
    conditions: list[dict] = []

    if parsed.categories:
        if len(parsed.categories) == 1:
            conditions.append({"category": parsed.categories[0]})
        else:
            conditions.append({"category": {"$in": parsed.categories}})

    if parsed.months:
        if len(parsed.months) == 1:
            conditions.append({"month": parsed.months[0]})
        else:
            conditions.append({"month": {"$in": parsed.months}})
    elif parsed.month_nums:
        if len(parsed.month_nums) == 1:
            conditions.append({"month_num": parsed.month_nums[0]})
        else:
            conditions.append({"month_num": {"$in": parsed.month_nums}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
