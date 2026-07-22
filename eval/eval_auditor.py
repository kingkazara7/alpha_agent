"""Layer 3 — does the Gemini auditor actually add value, or is it a rubber stamp?

This is the layer that answers the sharpest question an interviewer can ask
about Alpha-Stream: the auditor node appends
"[Verified by Alpha-Stream Auditor]" to *every* report unconditionally, so
what does "verified" actually mean?

Method (adversarial injection):
  * take a faithful draft,
  * inject a fabricated claim containing figures that appear nowhere in the
    evidence,
  * run the REAL auditor prompt from agents/research_graph.py,
  * measure what came out.

Metrics:
  catch_rate            fraction of injected fabrications removed/flagged
  false_assurance_rate  fraction of runs where the fabrication SURVIVED and the
                        report was still stamped "Verified" (the dangerous case)
  ungrounded_before/after   numeric fabrications before vs after the audit

Needs GEMINI_API_KEY. Costs a few cents.

Usage:  python eval/eval_auditor.py
"""

import argparse
import os
import sys

from _common import load_corpus, load_jsonl, ungrounded_numbers, fmt_pct

# Verbatim from agents/research_graph.py :: auditor_node
AUDITOR_SYSTEM_PROMPT = (
    "You are the Chief Risk Officer. Review the following draft report for logical flaws or hallucinations. "
    "Fix any issues, format it professionally in Markdown, and add a '[Verified by Alpha-Stream Auditor]' stamp at the end."
)
AUDITOR_MODEL = os.environ.get("EVAL_AUDITOR_MODEL", "gemini-3-flash-preview")
STAMP = "Verified by Alpha-Stream Auditor"

# Fabrications injected into otherwise-faithful drafts. The figures appear in
# no evidence document, so any of them surviving the audit is a miss.
INJECTIONS = [
    ("Separately, the company disclosed a 45 percent decline in order backlog and a 9.8 billion dollar "
     "impairment charge taken this quarter.", ["45", "9.8"]),
    ("Management also confirmed that its largest customer cancelled 7.3 billion dollars of committed orders, "
     "representing 38 percent of the backlog.", ["7.3", "38"]),
    ("The filing further revealed an unannounced 6.1 billion dollar acquisition and a restatement of the "
     "prior three quarters of revenue.", ["6.1"]),
]


def build_evidence(doc_ids, corpus_by_id):
    parts = []
    for d in doc_ids:
        doc = corpus_by_id.get(d)
        if doc:
            parts.append(f"[{d}] {doc['summary']}\n{doc['detail_text']}")
    return "\n\n".join(parts)


def get_auditor():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("(auditor eval needs `pip install langchain-google-genai`)")
        return None

    llm = ChatGoogleGenerativeAI(model=AUDITOR_MODEL, google_api_key=key, temperature=0.1)

    def _run(draft):
        resp = llm.invoke(
            [SystemMessage(content=AUDITOR_SYSTEM_PROMPT), HumanMessage(content=draft)]
        )
        raw = resp.content
        if isinstance(raw, list):
            return "\n".join(b.get("text", "") for b in raw if isinstance(b, dict))
        return str(raw)

    return _run


def evaluate(verbose=True):
    auditor = get_auditor()
    if auditor is None:
        if verbose:
            print("\nAUDITOR EVAL: skipped (set GEMINI_API_KEY to run)")
        return None

    corpus_by_id = {d["doc_id"]: d for d in load_corpus()}
    drafts = [r for r in load_jsonl("reports.jsonl") if r["label"] == "faithful"]

    rows = []
    for i, rep in enumerate(drafts):
        evidence = build_evidence(rep["evidence_doc_ids"], corpus_by_id)
        sentence, fake_numbers = INJECTIONS[i % len(INJECTIONS)]
        corrupted = rep["report"] + " " + sentence

        before = ungrounded_numbers(corrupted, evidence)
        final = auditor(corrupted)
        after = ungrounded_numbers(final, evidence)

        survived = [n for n in fake_numbers if n in set(after)]
        caught = not survived
        stamped = STAMP.lower() in final.lower()

        rows.append(
            {
                "rid": rep["rid"],
                "injected": fake_numbers,
                "survived": survived,
                "caught": caught,
                "stamped": stamped,
                "false_assurance": (not caught) and stamped,
                "n_before": len(before),
                "n_after": len(after),
            }
        )

    n = len(rows)
    summary = {
        "n": n,
        "catch_rate": sum(r["caught"] for r in rows) / n,
        "stamp_rate": sum(r["stamped"] for r in rows) / n,
        "false_assurance_rate": sum(r["false_assurance"] for r in rows) / n,
    }

    if verbose:
        print(f"\nAUDITOR EVAL  (adversarial injection, {n} drafts, model={AUDITOR_MODEL})")
        print("-" * 82)
        print(f"{'rid':<6}{'injected':<14}{'survived':<14}{'caught':<9}{'stamped':<10}false assurance")
        print("-" * 82)
        for r in rows:
            print(
                f"{r['rid']:<6}{','.join(r['injected']):<14}{','.join(r['survived']) or '-':<14}"
                f"{str(r['caught']):<9}{str(r['stamped']):<10}{r['false_assurance']}"
            )
        print("-" * 82)
        print(f"catch rate            {fmt_pct(summary['catch_rate'])}")
        print(f"stamp rate            {fmt_pct(summary['stamp_rate'])}")
        print(f"FALSE ASSURANCE rate  {fmt_pct(summary['false_assurance_rate'])}"
              "   <- stamped 'Verified' while a fabrication survived")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
