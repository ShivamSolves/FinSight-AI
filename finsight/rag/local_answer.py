"""
local_answer.py — Deterministic, offline answer generation from retrieved rows.

The RAG pipeline normally hands the retrieved transactions to an LLM to phrase
an answer. But the *numbers* don't need an LLM: when retrieval returns the
complete matching set, the correct total is just a sum. This module computes
that grounded answer locally so:

  • the demo works with no API key / no cost,
  • the UI stays responsive,
  • and there's a deterministic baseline to compare the LLM against.

It mirrors the answer format the LLM is instructed to use (Answer /
Transactions used / Confidence) so the UI can render either source uniformly.

Public API
──────────
  summarize_hits(hits, question, complete=False) -> LocalAnswer
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finsight.config import CURRENCY_SYMBOL as _S


@dataclass
class LocalAnswer:
    """A locally-computed grounded answer."""
    headline_total: float          # the dominant figure (spend, or income if no spend)
    spend_total:    float          # sum of |debits|
    income_total:   float          # sum of credits
    net:            float          # signed sum
    count:          int
    complete:       bool           # True if this was the full matching set
    evidence:       list[dict] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        return "High" if self.complete else "Medium"

    def transactions_md(self, limit: int | None = None) -> str:
        rows = self.evidence if limit is None else self.evidence[:limit]
        if not rows:
            return "_None_"
        lines = []
        for h in rows:
            m = h["metadata"]
            amt = abs(m["amount"])
            sign = "+" if m["transaction_type"] == "credit" else "−"
            lines.append(f"- `{m['date']}` · {m['description']} · {sign}{_S}{amt:,.2f} · _{m['category']}_")
        return "\n".join(lines)

    def to_markdown(self, question: str) -> str:
        """Render in the same shape the LLM prompt enforces."""
        parts = [f"**Answer:** {_S}{self.headline_total:,.2f} across {self.count} transaction"
                 f"{'s' if self.count != 1 else ''}."]
        if self.spend_total and self.income_total:
            parts.append(
                f"Spending {_S}{self.spend_total:,.2f} · Income {_S}{self.income_total:,.2f} · "
                f"Net {_S}{self.net:,.2f}."
            )
        parts.append("\n**Transactions used:**\n" + self.transactions_md())
        scope = "the complete matching set" if self.complete else "a relevance-ranked sample"
        parts.append(
            f"\n**Confidence:** {self.confidence} — computed directly from {scope} "
            f"of transactions (no estimate)."
        )
        return "\n".join(parts)


def summarize_hits(hits: list[dict], question: str = "", complete: bool = False) -> LocalAnswer:
    """
    Sum the retrieved transactions into a grounded LocalAnswer.

    Debits count as spending (absolute), credits as income. The headline figure
    is spending when there is any, otherwise income — which matches how people
    ask ("how much did I spend on X" vs "what was my income").
    """
    spend = income = net = 0.0
    for h in hits:
        amt = float(h["metadata"]["amount"])
        net += amt
        if amt < 0:
            spend += abs(amt)
        else:
            income += amt

    spend  = round(spend, 2)
    income = round(income, 2)
    net    = round(net, 2)
    headline = spend if spend > 0 else income

    # Sort evidence: biggest spend first, then credits, for readable output.
    ordered = sorted(hits, key=lambda h: h["metadata"]["amount"])

    return LocalAnswer(
        headline_total=headline,
        spend_total=spend,
        income_total=income,
        net=net,
        count=len(hits),
        complete=complete,
        evidence=ordered,
    )
