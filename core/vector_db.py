import os
import chromadb

# Ensure this points to the unified research_db folder in your project root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research_db")

class ResearchVault:
    def __init__(self):
        """
        Alpha-Stream's dedicated Retrieval Engine.
        Focuses on Level 1 (Macro Summaries) and Level 2 (Micro Details) hierarchical retrieval.
        """
        self.client = chromadb.PersistentClient(path=DB_PATH)
        # Must exactly match the collection name used in your document processor
        self.collection = self.client.get_or_create_collection(
            name="market_research",
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve_level_1_summaries(self, ticker: str, query: str, top_k: int = 3) -> list:
        """
        [Level 1 Retrieval: Token-Saving Mode]
        Searches only within 'Parent Nodes' (Summaries) to quickly locate relevant reports.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                # Core filter: Only search summaries, and only for the matching ticker
                where={
                    "$and": [
                        {"ticker": ticker},
                        {"chunk_type": "summary"}
                    ]
                }
            )
            
            summaries = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    summaries.append({
                        "doc_id": results['metadatas'][0][i]['doc_id'],
                        "summary_text": results['documents'][0][i],
                        "source": results['metadatas'][0][i].get('source', 'Unknown')
                    })
            return summaries
            
        except Exception as e:
            print(f"⚠️ Vector DB Query Warning: {e}")
            return []

    def retrieve_level_2_details(self, doc_id: str) -> str:
        """
        [Level 2 Retrieval: Deep Dive Mode]
        When the agent deems a summary critical, it uses the doc_id to pull all detailed child nodes.
        """
        results = self.collection.get(
            where={
                "$and": [
                    {"doc_id": doc_id},
                    {"chunk_type": "detail"}
                ]
            }
        )
        
        if not results['documents']:
            return "No detailed data chunks found."
            
        # Reconstruct the original text by sorting chunks by their original index
        chunks_with_index = zip(results['documents'], [meta['chunk_index'] for meta in results['metadatas']])
        sorted_chunks = sorted(chunks_with_index, key=lambda x: x[1])
        
        full_details = "\n...\n".join([chunk[0] for chunk in sorted_chunks])
        return full_details