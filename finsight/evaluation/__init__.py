"""
finsight.evaluation — offline retrieval evaluation + optional LLM-answer scoring.

    from finsight.evaluation import run_retrieval_evaluation, format_report
    report = run_retrieval_evaluation()
    print(format_report(report))
"""

from finsight.evaluation.dataset import GOLDEN_CASES, EvalCase, truth_total
from finsight.evaluation.metrics import (
    answer_matches,
    answer_total,
    extract_rupee_amounts,
    retrieval_scores,
)
from finsight.evaluation.runner import (
    CaseResult,
    EvalReport,
    format_report,
    run_answer_evaluation,
    run_retrieval_evaluation,
)

__all__ = [
    "GOLDEN_CASES",
    "EvalCase",
    "truth_total",
    "retrieval_scores",
    "answer_total",
    "answer_matches",
    "extract_rupee_amounts",
    "CaseResult",
    "EvalReport",
    "run_retrieval_evaluation",
    "run_answer_evaluation",
    "format_report",
]
