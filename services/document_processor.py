import uuid
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings

# Core modification: Point the path precisely to the existing research_db folder in the root directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "research_db")

class AlphaStreamIndexer:
    def __init__(self):
        # 1. Connect to the existing ChromaDB persistent directory
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="market_research")
        
        # 2. Configure local Ollama models (Ensure Ollama is running in the background)
        # Using llama3 for summarization and nomic-embed-text for vectorization
        self.llm = ChatOllama(model="llama3", temperature=0.1) 
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 3. Text Splitter configuration
        # Chunks of 500 tokens with a 50-token overlap to maintain semantic continuity
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )

    def process_and_store(self, raw_text: str, ticker: str, source_url: str = "local_test"):
        """Process raw long text, generate parent/child nodes, and store them in ChromaDB."""
        doc_id = str(uuid.uuid4())
        print(f"\n⏳ Starting to process data for [{ticker}], Doc ID: {doc_id}")

        # ==========================================
        # Step 1: Generate Parent Node (Summary Chunk)
        # ==========================================
        print("🧠 Calling LLM to generate core summary (Parent Node)...")
        prompt = ChatPromptTemplate.from_template(
            "You are a professional quantitative financial analyst. Based on the following news about {ticker}, "
            "extract the core events, supply chain risks, and market expectations that might affect its stock price. "
            "Limit your response to 300 words:\n\n{text}"
        )
        chain = prompt | self.llm
        
        # Process only the first 3000 characters to prevent context window overflow
        summary_text = chain.invoke({"ticker": ticker, "text": raw_text[:3000]}).content 
        
        # Store the summary in ChromaDB with 'chunk_type: summary' metadata
        self.collection.add(
            ids=[f"summary_{doc_id}"],
            embeddings=[self.embeddings.embed_query(summary_text)],
            documents=[summary_text],
            metadatas=[{"chunk_type": "summary", "doc_id": doc_id, "ticker": ticker, "source": source_url}]
        )
        print("✅ Parent node (Summary) generated and stored successfully!")

        # ==========================================
        # Step 2: Generate Child Nodes (Detail Chunks)
        # ==========================================
        print("🔪 Intelligently slicing the raw long text into chunks (Child Nodes)...")
        detail_chunks = self.splitter.split_text(raw_text)
        
        child_ids = [f"detail_{doc_id}_{i}" for i in range(len(detail_chunks))]
        child_metadatas = [
            {"chunk_type": "detail", "doc_id": doc_id, "ticker": ticker, "source": source_url, "chunk_index": i} 
            for i in range(len(detail_chunks))
        ]
        
        # Vectorize and store the detailed chunks in batch
        child_embeddings = self.embeddings.embed_documents(detail_chunks)
        self.collection.add(
            ids=child_ids,
            embeddings=child_embeddings,
            documents=detail_chunks,
            metadatas=child_metadatas
        )
        print(f"✅ Child nodes stored successfully! Sliced into {len(detail_chunks)} Detail Chunks.")
        print("-" * 50)


# ==========================================
# Test Entry Point
# ==========================================
if __name__ == "__main__":
    indexer = AlphaStreamIndexer()
    
    # Mock a lengthy $NVDA research report
    mock_nvda_report = """
    NVIDIA (NVDA) latest earnings report shows record-breaking revenue in its data center business, primarily driven by strong demand for the Hopper architecture H100 chips.
    However, supply chain insiders point out that the upcoming Blackwell architecture chips are facing severe capacity bottlenecks in TSMC's CoWoS packaging process.
    It is expected that this capacity constraint will have a slight impact on NVIDIA's gross margins in the first quarter of 2026.
    Meanwhile, AMD's newly launched MI300X accelerator has shown extreme cost-effectiveness in certain inference tasks, gradually eroding NVIDIA's market share among specific cloud computing clients.
    During the earnings call, management emphasized that the construction of next-generation AI factories will heavily rely on more efficient liquid cooling technologies, which will be the focus of future capital expenditures.
    """ * 10 # Multiply by 10 to simulate a long document and trigger the splitting logic
    
    indexer.process_and_store(raw_text=mock_nvda_report, ticker="NVDA")