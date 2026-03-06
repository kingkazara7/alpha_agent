import operator
import requests
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from core.vector_db import ResearchVault
from agents.notion_writer import NotionWriter
from core.config import settings

# ==========================================
# 1. Initialize LLMs via Settings
# ==========================================
# DeepSeek-V3.2 (Reasoner) via OpenAI compatible endpoint
deepseek_llm = ChatOpenAI(
    model="deepseek-reasoner", # Adjust based on your DeepSeek API tier/model name
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.2
)

# Gemini 3.1 Pro (Auditor)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.1
)

# ==========================================
# 2. Define the Graph State & Tools
# ==========================================
class AgentState(TypedDict):
    """
    State dictionary tracking the flow of data across the LangGraph nodes.
    """
    ticker: str
    market_event: str        # The real-time news/price trigger from WebSocket
    retrieved_context: str   # Summaries from local ChromaDB
    sec_data: str            # Official filings from SEC API
    draft_report: str        # DeepSeek's initial analysis
    final_report: str        # Gemini's audited output

# Initialize core tool instances
vault = ResearchVault()
notion_writer = NotionWriter()

# ==========================================
# 3. Define Graph Nodes (Agents & Tools)
# ==========================================
def retrieval_node(state: AgentState):
    """
    Node 1: Context Retrieval from local ChromaDB.
    """
    print(f"🔍 [Node: Retrieval] Fetching local context for {state['ticker']}...")
    summaries = vault.retrieve_level_1_summaries(
        ticker=state["ticker"], 
        query=state["market_event"]
    )
    context = "\n".join([f"- DocID {s['doc_id']}: {s['summary_text']}" for s in summaries])
    return {"retrieved_context": context if context else "No historical context found in local database."}


def sec_retrieval_node(state: AgentState):
    """
    Node 1.5: SEC Filings Retrieval.
    Fetches the latest official 8-K (Current Report) for the target ticker using sec-api.io.
    """
    print(f"🏛️ [Node: SEC] Fetching latest official SEC filings for {state['ticker']}...")
    
    # Query payload for sec-api.io (fetching the latest 1 filing)
    query = {
      "query": { "query_string": { "query": f"ticker:{state['ticker']} AND formType:\"8-K\"" } },
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" } }]
    }
    
    try:
        response = requests.post(
            f"https://api.sec-api.io?token={settings.SEC_API_KEY}",
            json=query,
            timeout=10 # Prevent hanging the pipeline
        )
        response.raise_for_status()
        filings = response.json().get('filings', [])
        
        if filings:
            latest_filing = filings[0]
            sec_context = f"Latest 8-K filed on {latest_filing.get('filedAt')}: {latest_filing.get('description', 'No description available')}"
        else:
            sec_context = "No recent 8-K filings found."
            
    except Exception as e:
        print(f"⚠️ [Warning] SEC API Fetch Failed: {e}")
        sec_context = "SEC data temporarily unavailable."
        
    return {"sec_data": sec_context}


def reasoner_node(state: AgentState):
    """
    Node 2: Strategy & Risk Analysis (DeepSeek).
    Synthesizes the real-time event, local history, and SEC data into a report.
    """
    print(f"🧠 [Node: Reasoner] DeepSeek is analyzing {state['ticker']} with combined data...")
    
    system_prompt = (
        "You are an elite quantitative researcher. Analyze the real-time market event, "
        "historical news context, and official SEC filings provided. "
        "Synthesize this data into a concise, professional strategy and risk assessment report."
    )
    human_prompt = (
        f"Event Trigger: {state['market_event']}\n\n"
        f"Official SEC Data:\n{state['sec_data']}\n\n"
        f"Historical Context (Internal DB):\n{state['retrieved_context']}"
    )
    
    response = deepseek_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    return {"draft_report": response.content}


def auditor_node(state: AgentState):
    """
    Node 3: Final Audit & Publishing (Gemini).
    Reviews the draft, ensures quality, formats it, and publishes to Notion.
    """
    print(f"⚖️ [Node: Auditor] Gemini is verifying the draft and publishing...")
    
    system_prompt = (
        "You are the Chief Risk Officer. Review the following draft report for logical flaws or hallucinations. "
        "Fix any issues, format it professionally in Markdown, and add a '[Verified by Alpha-Stream Auditor]' stamp at the end."
    )
    
    response = gemini_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['draft_report'])
    ])
    
    raw_content = response.content
    if isinstance(raw_content, list):
        final_report = "\n".join([block.get("text", "") for block in raw_content if isinstance(block, dict)])
    else:
        final_report = str(raw_content)
    
    # Trigger the Notion Writer
    report_title = f"Alpha-Stream Alert: {state['ticker']} Analysis"
    try:
        notion_writer.write(title=report_title, content=final_report)
    except Exception as e:
        print(f"❌ [Error] Notion Publishing Failed: {e}")
        
    return {"final_report": final_report}

# ==========================================
# 4. Compile the LangGraph
# ==========================================
def build_research_graph():
    """
    Constructs the directed graph defining the multi-agent workflow.
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("sec_retriever", sec_retrieval_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("auditor", auditor_node)

    # Define the execution edges (The pipeline flow)
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "sec_retriever") 
    workflow.add_edge("sec_retriever", "reasoner")
    workflow.add_edge("reasoner", "auditor")
    workflow.add_edge("auditor", END)

    return workflow.compile()

# Instantiate the compiled graph for external imports (used by watcher.py)
alpha_stream_app = build_research_graph()

# ==========================================
# Local Test Execution
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Local Alpha-Stream Graph Test...")
    
    # Mock input to test the pipeline without WebSocket
    test_inputs = {
        "ticker": "NVDA",
        "market_event": "Abnormal trade volume detected. Current Price: $140.50, Volume: 150000."
    }
    
    try:
        for output in alpha_stream_app.stream(test_inputs):
            for key, value in output.items():
                print(f"--- Finished Node: {key} ---")
        print("\n✅ Local Execution Complete! Check your Notion Database.")
    except Exception as e:
        print(f"\n❌ Pipeline failed during local test: {e}")