"""Layer 1 — retrieval eval. No API keys, no LLM, no cost.

Measures whether Level-1 summary retrieval actually surfaces the document a
query is about. This is the most objective layer and the right thing to gate
CI on: if retrieval misses, every downstream generation metric is measuring
the wrong thing.

Metrics (per query, then averaged):
  hit@k        did at least one gold doc appear in the top k
  recall@k     fraction of gold docs retrieved
  precision@k  fraction of retrieved docs that are gold
  MRR          1 / rank of the first gold doc (0 if absent)

Also checks that Level-2 drill-down returns detail text for each gold doc,
because in production `retrieve_level_2_details` is defined but never called.

Usage:  python eval/eval_retrieval.py [--k 3]
"""

import argparse
import sys

from _common import EvalVault, load_queries, fmt_pct


def evaluate(top_k=3, verbose=True):
    vault = EvalVault()
    queries = load_queries()

    hits, recalls, precisions, rrs = [], [], [], []
    l2_ok, l2_total = 0, 0
    rows = []

    for q in queries:
        gold = set(q["gold_doc_ids"])
        retrieved = vault.retrieve_level_1_summaries(q["ticker"], q["event"], top_k=top_k)
        got = [r["doc_id"] for r in retrieved]

        inter = gold.intersection(got)
        hit = 1.0 if inter else 0.0
        recall = len(inter) / len(gold) if gold else 0.0
        precision = len(inter) / len(got) if got else 0.0
        rr = 0.0
        for rank, d in enumerate(got, start=1):
            if d in gold:
                rr = 1.0 / rank
                break

        hits.append(hit)
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)
        rows.append((q["qid"], q["ticker"], hit, recall, rr, got))

        # Level-2 drill-down sanity: every gold doc must yield detail text
        for d in gold:
            l2_total += 1
            if vault.retrieve_level_2_details(d).strip():
                l2_ok += 1

    n = len(queries)
    summary = {
        "n_queries": n,
        "top_k": top_k,
        "hit_at_k": sum(hits) / n,
        "recall_at_k": sum(recalls) / n,
        "precision_at_k": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "l2_drilldown_ok": l2_ok / l2_total if l2_total else 0.0,
    }

    if verbose:
        print(f"\nRETRIEVAL EVAL  (top_k={top_k}, {n} queries)")
        print("-" * 74)
        print(f"{'qid':<5}{'ticker':<8}{'hit':<6}{'recall':<9}{'RR':<7}retrieved")
        print("-" * 74)
        for qid, tic, hit, rec, rr, got in rows:
            mark = "OK " if hit else "MISS"
            print(f"{qid:<5}{tic:<8}{mark:<6}{rec:<9.2f}{rr:<7.2f}{','.join(got) or '(none)'}")
        print("-" * 74)
        print(f"hit@{top_k}      {fmt_pct(summary['hit_at_k'])}")
        print(f"recall@{top_k}   {fmt_pct(summary['recall_at_k'])}")
        print(f"precision@{top_k}{fmt_pct(summary['precision_at_k'])}")
        print(f"MRR         {summary['mrr']:.3f}")
        print(f"L2 drill-down returns detail text: {fmt_pct(summary['l2_drilldown_ok'])}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    try:
        evaluate(top_k=args.k)
    except Exception as exc:  # collection missing, etc.
        print(f"retrieval eval could not run: {exc}")
        print("did you run `python eval/seed_corpus.py` first?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
