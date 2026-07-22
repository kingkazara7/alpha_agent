"""Build the eval Chroma database from corpus.jsonl.

Mirrors the production schema written by services/document_processor.py
(summary parent node + detail child nodes, same metadata keys), but lets Chroma
compute embeddings with its default function on BOTH sides so that
`query_texts=` retrieval works. See _common.py for the production bug this
sidesteps.

Usage:  python eval/seed_corpus.py
"""

import shutil
import sys

from _common import EVAL_DB_PATH, load_corpus, split_text, get_collection


def main():
    # start clean so the eval is reproducible
    shutil.rmtree(EVAL_DB_PATH, ignore_errors=True)
    collection = get_collection(create=True)

    corpus = load_corpus()
    n_summary = n_detail = 0

    for doc in corpus:
        doc_id = doc["doc_id"]

        # --- parent node: the summary (what Level-1 retrieval searches) ---
        collection.add(
            ids=[f"summary_{doc_id}"],
            documents=[doc["summary"]],
            metadatas=[
                {
                    "chunk_type": "summary",
                    "doc_id": doc_id,
                    "ticker": doc["ticker"],
                    "source": doc["source"],
                }
            ],
        )
        n_summary += 1

        # --- child nodes: the detail chunks (what Level-2 drill-down pulls) ---
        chunks = split_text(doc["detail_text"], chunk_size=500, overlap=50)
        collection.add(
            ids=[f"detail_{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[
                {
                    "chunk_type": "detail",
                    "doc_id": doc_id,
                    "ticker": doc["ticker"],
                    "source": doc["source"],
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ],
        )
        n_detail += len(chunks)

    print(f"seeded {len(corpus)} documents -> {n_summary} summaries, {n_detail} detail chunks")
    print(f"eval db: {EVAL_DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
