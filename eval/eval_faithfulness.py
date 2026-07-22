"""Layer 2 — faithfulness / groundedness eval.

In finance the dangerous hallucination is a fabricated figure, so this layer
has two halves:

  1. DETERMINISTIC (always runs, no API key, no cost)
     Every numeric token in the report must appear in the evidence.
     A rule catches this more reliably than an LLM judge.

  2. LLM-AS-JUDGE (needs an API key)
     Decompose the report into atomic factual claims, then verify each one
     against ONLY the retrieved evidence -> supported / unsupported /
     contradicted.  faithfulness = supported / total.

     The judge is deliberately a *different* model family than the generator
     (DeepSeek) and the auditor (Gemini) to avoid self-preference bias.

reports.jsonl contains one deliberately hallucinated report (r04) so you can
confirm the metric actually discriminates rather than rubber-stamping.

Usage:
  python eval/eval_faithfulness.py              # rule check only if no key
  python eval/eval_faithfulness.py --judge      # also run the LLM judge
"""

import argparse
import json
import os
import sys

from _common import load_corpus, load_jsonl, ungrounded_numbers, fmt_pct

JUDGE_PROMPT = """You are a strict fact-checking judge.

You are given EVIDENCE and a CLAIM extracted from an analyst report.
Decide whether the CLAIM is supported by the EVIDENCE **alone**.
Do not use outside knowledge. If the evidence does not state it, it is not supported.

Answer with exactly one word: SUPPORTED, UNSUPPORTED, or CONTRADICTED.

EVIDENCE:
{evidence}

CLAIM:
{claim}

Answer:"""

DECOMPOSE_PROMPT = """Break the following analyst report into atomic factual claims.
Each claim must be a single, self-contained, checkable statement.
Ignore pure opinion/recommendation sentences.
Return one claim per line, no numbering, no extra text.

REPORT:
{report}"""


def build_evidence(doc_ids, corpus_by_id):
    parts = []
    for d in doc_ids:
        doc = corpus_by_id.get(d)
        if doc:
            parts.append(f"[{d}] {doc['summary']}\n{doc['detail_text']}")
    return "\n\n".join(parts)


def get_judge():
    """Return a callable(prompt)->str, or None if no key is configured."""
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("(judge needs `pip install openai`)")
        return None
    client = OpenAI(api_key=key)

    def _call(prompt):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


def evaluate(use_judge=False, verbose=True):
    corpus_by_id = {d["doc_id"]: d for d in load_corpus()}
    reports = load_jsonl("reports.jsonl")
    judge = get_judge() if use_judge else None

    results = []
    for rep in reports:
        evidence = build_evidence(rep["evidence_doc_ids"], corpus_by_id)
        bad_numbers = ungrounded_numbers(rep["report"], evidence)

        row = {
            "rid": rep["rid"],
            "label": rep["label"],
            "ungrounded_numbers": bad_numbers,
            "number_check_pass": not bad_numbers,
            "faithfulness": None,
            "n_claims": None,
        }

        if judge:
            claims = [
                c.strip("-• ").strip()
                for c in judge(DECOMPOSE_PROMPT.format(report=rep["report"])).splitlines()
                if c.strip()
            ]
            verdicts = []
            for c in claims:
                v = judge(JUDGE_PROMPT.format(evidence=evidence, claim=c)).upper()
                verdicts.append("SUPPORTED" if v.startswith("SUPPORTED") else v.split()[0] if v else "UNSUPPORTED")
            supported = sum(1 for v in verdicts if v == "SUPPORTED")
            row["faithfulness"] = supported / len(claims) if claims else 0.0
            row["n_claims"] = len(claims)

        results.append(row)

    if verbose:
        print("\nFAITHFULNESS EVAL")
        print("-" * 78)
        print(f"{'rid':<6}{'label':<15}{'numbers OK':<12}{'faithfulness':<14}ungrounded numbers")
        print("-" * 78)
        for r in results:
            f = "n/a" if r["faithfulness"] is None else f"{r['faithfulness']:.2f} ({r['n_claims']})"
            print(
                f"{r['rid']:<6}{r['label']:<15}{str(r['number_check_pass']):<12}{f:<14}"
                f"{','.join(r['ungrounded_numbers']) or '-'}"
            )
        print("-" * 78)
        faithful = [r for r in results if r["label"] == "faithful"]
        halluc = [r for r in results if r["label"] == "hallucinated"]
        print(f"rule check: {sum(r['number_check_pass'] for r in faithful)}/{len(faithful)} faithful reports pass, "
              f"{sum(not r['number_check_pass'] for r in halluc)}/{len(halluc)} hallucinated reports caught")
        if judge:
            for grp, name in ((faithful, "faithful"), (halluc, "hallucinated")):
                vals = [r["faithfulness"] for r in grp if r["faithfulness"] is not None]
                if vals:
                    print(f"mean faithfulness ({name}): {sum(vals)/len(vals):.2f}")
        else:
            print("judge not run (set OPENAI_API_KEY and pass --judge for the LLM half)")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the LLM-as-judge half")
    args = ap.parse_args()
    evaluate(use_judge=args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
