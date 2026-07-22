# Alpha-Stream eval suite

An evaluation harness for the LangGraph research pipeline
(`retriever → sec_retriever → reasoner (DeepSeek) → auditor (Gemini)`).

Before this existed the pipeline had **no evaluation at all** — the only quality
mechanism was an inline Gemini "auditor" node. This suite replaces "the demo
looks good" with numbers, in three layers, cheapest and most objective first.

```
eval/
├── corpus.jsonl        8 labelled financial documents (summary + detail text)
├── queries.jsonl       10 labelled market-event queries with gold doc ids
├── reports.jsonl       4 fixture reports — 3 faithful, 1 deliberately hallucinated
├── seed_corpus.py      builds eval_db/ (isolated from production research_db/)
├── eval_retrieval.py   Layer 1 — retrieval metrics          (no API key)
├── eval_faithfulness.py Layer 2 — groundedness              (rule half free)
├── eval_auditor.py     Layer 3 — is the auditor a rubber stamp?  (needs key)
└── run_all.py          runs everything available, prints a summary
```

## Quick start

```bash
pip install chromadb
python eval/seed_corpus.py     # one-off: build the eval vector store
python eval/run_all.py         # free layers only
python eval/run_all.py --judge # + LLM-as-judge and the auditor injection test
```

Environment variables used by the optional layers:
`OPENAI_API_KEY` (judge), `GEMINI_API_KEY` (auditor).
Optional overrides: `EVAL_JUDGE_MODEL`, `EVAL_AUDITOR_MODEL`.

## The three layers

### Layer 1 — retrieval (`eval_retrieval.py`)
Given a labelled `(ticker, market event) → gold doc_ids` set, measures
**hit@k, recall@k, precision@k, MRR** on Level-1 summary retrieval, and
verifies Level-2 drill-down returns detail text for every gold document.

This is the layer to gate CI on: it is deterministic, free, and a generation
failure is very often really a retrieval failure.

Current result on the seed set (k=3):

| metric | value |
|---|---|
| hit@3 | 100% |
| recall@3 | 100% |
| precision@3 | 70% |
| MRR | 1.000 |
| L2 drill-down returns text | 100% |

> Honest framing: the seed corpus is small (8 docs) and queries are filtered by
> ticker, so perfect recall is expected. This is a **regression gate and a proof
> that the retrieval path works when embeddings are consistent** — not a hard
> benchmark. Grow `corpus.jsonl` / `queries.jsonl` to make it discriminative.

### Layer 2 — faithfulness (`eval_faithfulness.py`)
Two halves:

1. **Deterministic (always runs).** Every numeric token in the report must
   appear in the evidence. In finance a fabricated figure is the most dangerous
   hallucination, and a rule catches it more reliably than a judge.
2. **LLM-as-judge (`--judge`).** Decompose the report into atomic claims, then
   verify each against the retrieved evidence only →
   `faithfulness = supported / total`. The judge is a *different* model family
   from the generator (DeepSeek) and the auditor (Gemini) to avoid
   self-preference bias.

`reports.jsonl` includes one deliberately hallucinated report so you can confirm
the metric discriminates instead of passing everything.

Current result (rule half): **3/3 faithful reports pass, 1/1 hallucinated caught**
(flags the fabricated `31.7`, `35.0`, `40`, `47`).

### Layer 3 — auditor value-add (`eval_auditor.py`)
The pointed question about this pipeline: `auditor_node` appends
`[Verified by Alpha-Stream Auditor]` to **every** report unconditionally, so the
stamp asserts a guarantee nothing checks.

This layer injects a fabricated claim into an otherwise faithful draft, runs the
**real** auditor prompt, and measures:

- `catch_rate` — fabrications removed or flagged
- `stamp_rate` — how often the "Verified" stamp is applied
- `false_assurance_rate` — **stamped "Verified" while the fabrication survived**

That last number is the one that matters. Needs `GEMINI_API_KEY`.

## Two production bugs this suite exposed

1. **The production vector store is empty.** `research_db/chroma.sqlite3` has
   0 embeddings, so `retrieval_node` always falls through to
   *"No historical context found in local database."*

2. **Embedding dimension mismatch (silent).**
   `services/document_processor.py` writes vectors from Ollama
   `nomic-embed-text` (768-dim) via an explicit `embeddings=[...]` argument,
   while `core/vector_db.py` queries with `query_texts=[...]`, which makes Chroma
   embed the query with its *default* function (all-MiniLM, 384-dim). The
   dimensions do not match, the query raises, and the `try/except` in
   `retrieve_level_1_summaries` swallows it and returns `[]` — retrieval fails
   silently.

   **Fix:** pass the same `embedding_function` when creating the collection on
   both sides, or embed queries with the same Ollama model used at ingest.
   This suite sidesteps it by using Chroma's default function consistently in
   its own isolated database.

## Suggested next steps

- Wire Layer 1 + the Layer 2 rule half into `.github/workflows/` (currently empty)
  as a required check — both are free and deterministic.
- Grow the labelled set; add negative queries (a ticker with no relevant doc) to
  test abstention rather than forced top-k.
- Add `answer_relevance` and `context_relevance` (the rest of the RAG triad).
- Replace the unconditional stamp with a structured auditor verdict
  (`pass` / `flagged` + reasons) once `false_assurance_rate` is measured.
