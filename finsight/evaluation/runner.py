"""
evaluation/runner.py — Orchestrates the evaluation and reports the results.

Two independent layers:

  • run_retrieval_evaluation() — OFFLINE. Builds the store and checks that the
    retriever pulls the complete, correct set for every golden case (recall,
    precision, total accuracy). Needs no API key. This is the layer that guards
    the correctness of aggregate answers.

  • run_answer_evaluation() — ONLINE. Calls the real LLM via qa_pipeline.ask()
    and checks the ₹ total in its reply against ground truth. Auto-skips when
    OPENAI_API_KEY is absent, so the suite never fails just for lack of a key.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from finsight.categorization.categorizer import categorize_dataframe
from finsight.config import RAW_DIR
from finsight.embeddings.vector_store import build_vector_store
from finsight.evaluation.dataset import GOLDEN_CASES, EvalCase, truth_total
from finsight.evaluation.metrics import answer_total, retrieval_scores
from finsight.ingestion.csv_extractor import load_csv
from finsight.rag.qa_pipeline import ask, retrieve

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    id:              str
    label:           str
    question:        str
    n_expected:      int
    n_retrieved:     int
    recall:          float
    precision:       float
    expected_total:  float
    retrieved_total: float
    total_ok:        bool
    answer_total:    float | None = None   # only set by the LLM evaluation
    answer_ok:       bool | None = None

    @property
    def passed(self) -> bool:
        # Retrieval must be complete and exact; if an LLM answer was scored it
        # must also match (answer_ok is None for the offline retrieval layer).
        return self.recall >= 0.999 and self.total_ok and self.answer_ok is not False


@dataclass
class EvalReport:
    cases:   list[CaseResult] = field(default_factory=list)
    layer:   str = "retrieval"

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def ok(self) -> bool:
        return self.passed == self.total

    @property
    def mean_recall(self) -> float:
        return round(sum(c.recall for c in self.cases) / self.total, 4) if self.total else 0.0


def _prepared_dataframe(csv_path=None):
    """Load + categorize the statement (the input the store is built from)."""
    return categorize_dataframe(load_csv(csv_path or RAW_DIR / "bank_statement.csv"))


def run_retrieval_evaluation(cases: list[EvalCase] = GOLDEN_CASES,
                             csv_path=None, rebuild: bool = True) -> EvalReport:
    """Offline retrieval evaluation. Builds the store once, scores every case."""
    df = _prepared_dataframe(csv_path)
    if rebuild:
        build_vector_store(df)

    report = EvalReport(layer="retrieval")
    for case in cases:
        hits, _parsed, _complete = retrieve(case.question)
        s = retrieval_scores(df, hits, case)
        report.cases.append(CaseResult(
            id=case.id, label=case.label, question=case.question,
            n_expected=s["n_expected"], n_retrieved=s["n_retrieved"],
            recall=s["recall"], precision=s["precision"],
            expected_total=s["expected_total"], retrieved_total=s["retrieved_total"],
            total_ok=s["total_ok"],
        ))
    return report


def run_answer_evaluation(cases: list[EvalCase] = GOLDEN_CASES,
                          csv_path=None) -> EvalReport | None:
    """
    Online LLM-answer evaluation. Returns None (skipped) when no API key is set.
    Assumes the vector store is already built.
    """
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set — skipping answer evaluation.")
        return None

    df = _prepared_dataframe(csv_path)
    report = EvalReport(layer="answer")
    for case in cases:
        hits, _parsed, _complete = retrieve(case.question)
        s = retrieval_scores(df, hits, case)
        result = ask(case.question)
        ans_total = answer_total(result.answer)
        expected = truth_total(df, case)
        report.cases.append(CaseResult(
            id=case.id, label=case.label, question=case.question,
            n_expected=s["n_expected"], n_retrieved=s["n_retrieved"],
            recall=s["recall"], precision=s["precision"],
            expected_total=expected, retrieved_total=s["retrieved_total"],
            total_ok=s["total_ok"],
            answer_total=ans_total,
            answer_ok=(ans_total is not None and abs(ans_total - expected) < 0.01),
        ))
    return report


def format_report(report: EvalReport) -> str:
    """Render an EvalReport as a readable table."""
    sep = "─" * 78
    lines = [sep, f"  EVALUATION — {report.layer.upper()}", sep]
    for c in report.cases:
        flag = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{flag}] {c.label}")
        lines.append(
            f"         recall={c.recall:.2f} precision={c.precision:.2f} | "
            f"retrieved {c.n_retrieved}/{c.n_expected} txns | "
            f"₹{c.retrieved_total:,.2f} / expected ₹{c.expected_total:,.2f}"
        )
        if c.answer_total is not None:
            aflag = "✓" if c.answer_ok else "✗"
            lines.append(f"         LLM answer total: ₹{c.answer_total:,.2f} {aflag}")
    lines.append(sep)
    lines.append(
        f"  {report.passed}/{report.total} cases passed | "
        f"mean recall {report.mean_recall:.3f}"
    )
    lines.append(sep)
    return "\n".join(lines)
