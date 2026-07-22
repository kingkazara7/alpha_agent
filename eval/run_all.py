"""Run every eval layer that can run in the current environment and print a summary.

Layers that need no API key always run; layers that need one are skipped with a
note rather than failing, so this is safe to wire into CI.

Usage:
  python eval/run_all.py            # rule-based layers only
  python eval/run_all.py --judge    # also the LLM-as-judge + auditor layers
"""

import argparse
import os
import sys

import eval_retrieval
import eval_faithfulness
import eval_auditor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="also run the layers that call external LLM APIs")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    print("=" * 82)
    print("ALPHA-STREAM EVAL SUITE")
    print("=" * 82)

    # ---- Layer 1: retrieval (free) ----
    try:
        retrieval = eval_retrieval.evaluate(top_k=args.k)
    except Exception as exc:
        print(f"\nRETRIEVAL EVAL failed: {exc}\nrun `python eval/seed_corpus.py` first")
        retrieval = None

    # ---- Layer 2: faithfulness (rule half free, judge half optional) ----
    faith = eval_faithfulness.evaluate(use_judge=args.judge)

    # ---- Layer 3: auditor value-add (needs GEMINI_API_KEY) ----
    auditor = eval_auditor.evaluate() if args.judge else None
    if not args.judge:
        print("\nAUDITOR EVAL: skipped (pass --judge to run)")

    # ---- summary ----
    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    if retrieval:
        print(f"retrieval   hit@{retrieval['top_k']}={retrieval['hit_at_k']:.2f}  "
              f"recall@{retrieval['top_k']}={retrieval['recall_at_k']:.2f}  "
              f"MRR={retrieval['mrr']:.3f}  L2={retrieval['l2_drilldown_ok']:.2f}")
    if faith:
        caught = sum(1 for r in faith if r["label"] == "hallucinated" and not r["number_check_pass"])
        total_h = sum(1 for r in faith if r["label"] == "hallucinated")
        clean = sum(1 for r in faith if r["label"] == "faithful" and r["number_check_pass"])
        total_f = sum(1 for r in faith if r["label"] == "faithful")
        print(f"faithfulness  number-rule: {clean}/{total_f} faithful pass, {caught}/{total_h} hallucinated caught")
    if auditor:
        print(f"auditor     catch={auditor['catch_rate']:.2f}  "
              f"stamp={auditor['stamp_rate']:.2f}  "
              f"false-assurance={auditor['false_assurance_rate']:.2f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
