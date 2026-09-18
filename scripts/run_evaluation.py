"""
run_evaluation.py — CLI for the FinSight evaluation harness.

By default this runs the OFFLINE retrieval evaluation: it builds the vector
store and checks that every golden question retrieves the complete, correct set
of transactions (recall, precision, exact total). No API key required.

Add --llm to also call the real LLM and score the ₹ total in its answer
(requires OPENAI_API_KEY; skipped automatically when the key is absent).

Run:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --llm
    python scripts/run_evaluation.py --csv data/raw/my_statement.csv

Exit code is non-zero if any case fails, so this doubles as a CI gate.
"""

import argparse
import logging
import sys

from finsight.evaluation import (
    format_report,
    run_answer_evaluation,
    run_retrieval_evaluation,
)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="FinSight AI — evaluation harness")
    parser.add_argument("--csv", default=None, help="Path to a bank statement CSV (default: data/raw/bank_statement.csv)")
    parser.add_argument("--llm", action="store_true", help="Also score real LLM answers (needs OPENAI_API_KEY)")
    args = parser.parse_args()

    print("\nRunning offline retrieval evaluation...")
    report = run_retrieval_evaluation(csv_path=args.csv)
    print(format_report(report))
    exit_code = 0 if report.ok else 1

    if args.llm:
        print("\nRunning LLM answer evaluation...")
        ans_report = run_answer_evaluation(csv_path=args.csv)
        if ans_report is None:
            print("  Skipped — OPENAI_API_KEY not set.")
        else:
            print(format_report(ans_report))
            if not ans_report.ok:
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
