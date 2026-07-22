"""Shared helpers for the Alpha-Stream eval suite.

The eval suite deliberately uses its OWN Chroma database (eval/eval_db) so it
never touches the production research_db, and it uses Chroma's default
embedding function for BOTH writes and queries so that retrieval actually
works end to end.

NOTE on a production bug this suite exposes: services/document_processor.py
writes vectors computed by Ollama `nomic-embed-text` (768-dim) via an explicit
`embeddings=[...]` argument, while core/vector_db.py queries with
`query_texts=[...]`, which makes Chroma embed the query with its *default*
function (all-MiniLM, 384-dim). The dimensions do not match, so the query
raises, gets swallowed by the try/except in retrieve_level_1_summaries, and
retrieval silently returns []. Either both sides must use the same embedding
function, or the collection must be created with an explicit
`embedding_function`.
"""

import json
import os
import re

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DB_PATH = os.path.join(EVAL_DIR, "eval_db")
COLLECTION_NAME = "market_research_eval"


def load_jsonl(filename):
    path = os.path.join(EVAL_DIR, filename)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus():
    return load_jsonl("corpus.jsonl")


def load_queries():
    return load_jsonl("queries.jsonl")


def split_text(text, chunk_size=500, overlap=50):
    """Minimal recursive-ish splitter (mirrors the 500/50 config in production)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # prefer to break on a sentence boundary, else a space
        window = text[start:end]
        cut = max(window.rfind(". "), window.rfind("\n"))
        if cut < chunk_size * 0.5:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = chunk_size
        else:
            cut += 1
        chunks.append(text[start:start + cut].strip())
        start += max(cut - overlap, 1)
    return [c for c in chunks if c]


def get_collection(create=False):
    import chromadb

    client = chromadb.PersistentClient(path=EVAL_DB_PATH)
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return client.get_collection(name=COLLECTION_NAME)


class EvalVault:
    """Mirrors core/vector_db.ResearchVault against the eval database."""

    def __init__(self):
        self.collection = get_collection()

    def retrieve_level_1_summaries(self, ticker, query, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"$and": [{"ticker": ticker}, {"chunk_type": "summary"}]},
        )
        out = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                out.append(
                    {
                        "doc_id": results["metadatas"][0][i]["doc_id"],
                        "summary_text": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                    }
                )
        return out

    def retrieve_level_2_details(self, doc_id):
        results = self.collection.get(
            where={"$and": [{"doc_id": doc_id}, {"chunk_type": "detail"}]}
        )
        if not results["documents"]:
            return ""
        pairs = zip(results["documents"], [m["chunk_index"] for m in results["metadatas"]])
        return "\n...\n".join(c for c, _ in sorted(pairs, key=lambda x: x[1]))


# ---------------------------------------------------------------- numbers ---

NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def extract_numbers(text):
    """All numeric tokens in a piece of text, normalised (commas stripped)."""
    return {m.group(0).replace(",", "").rstrip(".") for m in NUMBER_RE.finditer(text or "")}


def ungrounded_numbers(report, evidence):
    """Numbers that appear in the report but in none of the evidence.

    This is the deterministic half of the faithfulness check: in finance a
    fabricated figure is the most dangerous hallucination, and a rule catches
    it more reliably than an LLM judge.
    """
    ev = extract_numbers(evidence)
    bad = []
    for n in sorted(extract_numbers(report)):
        # ignore trivial tokens (years, single digits, list markers)
        if len(n) <= 1:
            continue
        if n in ev:
            continue
        # tolerate a figure that appears with different precision, e.g. 26.3 vs 26.30
        if any(n.rstrip("0").rstrip(".") == e.rstrip("0").rstrip(".") for e in ev):
            continue
        bad.append(n)
    return bad


def fmt_pct(x):
    return f"{100 * x:5.1f}%"
