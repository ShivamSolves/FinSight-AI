"""
evaluation/metrics.py — Scoring functions for retrieval and answer quality.

Retrieval metrics compare the set of transactions the pipeline pulled against
the set the CSV says it *should* have pulled:

    recall    = |retrieved ∩ expected| / |expected|   ← did we find everything?
    precision = |retrieved ∩ expected| / |retrieved|  ← did we pull in noise?

For aggregate questions recall is the number that matters: a recall < 1.0 means
rows were dropped and any total computed from them is wrong.

Answer metrics parse the ₹ total out of the LLM's reply and compare it to ground
truth — used only when an API key is available.
"""

from __future__ import annotations

import re

from finsight.evaluation.dataset import EvalCase, expected_keys, row_key, truth_total

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieved_keys(hits: list[dict]) -> set[tuple]:
    """Identity set of the transactions a retrieval call returned."""
    return {
        row_key(h["metadata"]["date"], h["metadata"]["description"], h["metadata"]["amount"])
        for h in hits
    }


def retrieval_scores(df, hits: list[dict], case: EvalCase) -> dict:
    """Compute recall, precision, and total accuracy for one case."""
    exp = expected_keys(df, case)
    got = retrieved_keys(hits)

    intersection = exp & got
    recall    = len(intersection) / len(exp) if exp else 1.0
    precision = len(intersection) / len(got) if got else (1.0 if not exp else 0.0)

    retrieved_total = round(sum(abs(h["metadata"]["amount"]) for h in hits), 2)
    expected_total  = truth_total(df, case)

    return {
        "n_expected":      len(exp),
        "n_retrieved":     len(got),
        "recall":          round(recall, 4),
        "precision":       round(precision, 4),
        "retrieved_total": retrieved_total,
        "expected_total":  expected_total,
        "total_ok":        abs(retrieved_total - expected_total) < 0.01,
        "hardcoded_total": case.expected_total,
        "matches_hardcoded": abs(expected_total - case.expected_total) < 0.01,
    }


# ── Answer parsing ────────────────────────────────────────────────────────────

_RUPEE_RE = re.compile(r"₹\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")


def extract_rupee_amounts(text: str) -> list[float]:
    """Pull every ₹-formatted number out of an answer string."""
    return [float(m.replace(",", "")) for m in _RUPEE_RE.findall(text or "")]


def answer_total(answer: str) -> float | None:
    """
    Best-effort single total from an LLM answer.

    The system prompt puts the answer on the '**Answer:**' line, so we prefer the
    first ₹ amount found there; otherwise we take the largest amount mentioned
    (totals dominate over individual line items).
    """
    if not answer:
        return None

    for line in answer.splitlines():
        if line.strip().lower().startswith("**answer"):
            amounts = extract_rupee_amounts(line)
            if amounts:
                return amounts[0]

    amounts = extract_rupee_amounts(answer)
    return max(amounts) if amounts else None


def answer_matches(answer: str, expected: float, tol: float = 0.01) -> bool:
    total = answer_total(answer)
    return total is not None and abs(total - expected) <= tol
