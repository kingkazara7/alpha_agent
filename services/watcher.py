import websocket
import json
import threading
import os
from core.config import settings
from agents.research_graph import alpha_stream_app

# The path to your external configuration file
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "watch_config.json")

def load_watch_config() -> dict:
    """
    [Hot-Reloading Logic]
    Reads the external JSON file. If the file is missing or broken, 
    it returns a safe default list to prevent the system from crashing.
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not read {CONFIG_FILE}. Using empty defaults. Error: {e}")
        return {"tickers": [], "keywords": []}

def trigger_graph_pipeline(trigger_reason: str, ticker: str, headline: str, summary: str):
    """Executes the Alpha-Stream LangGraph in a separate thread."""
    market_event = f"Trigger: {trigger_reason}\nBreaking News: {headline}\nSummary: {summary}"
    # If it's a macro keyword, we can use a proxy ticker like 'SPY' for the analysis context
    analysis_ticker = ticker if ticker else "SPY" 
    
    print(f"\n⚡ [ALERT] Triggering Alpha-Stream Graph for {analysis_ticker} ({trigger_reason})!")
    
    inputs = {
        "ticker": analysis_ticker,
        "market_event": market_event
    }
    
    try:
        for output in alpha_stream_app.stream(inputs):
            pass # Suppress graph node printouts for cleaner news stream
        print(f"✅ Full Pipeline finished. Report pushed to Notion for {analysis_ticker}!")
    except Exception as e:
        print(f"❌ Pipeline Execution Failed: {e}")

def on_message(ws, message):
    data = json.loads(message)
    
    for msg in data:
        if msg.get('T') == 'success' and msg.get('msg') == 'authenticated':
            print("🔐 Authenticated! Subscribing to ALL news for local filtering...")
            # We subscribe to "*" (All News) to catch macro keywords
            ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
            
        elif msg.get('T') == 'n':
            symbols = msg.get('symbols', [])
            headline = msg.get('headline', '')
            summary = msg.get('summary', '')
            
            # 1. Hot-reload the config file (reads the latest saved state instantly)
            config = load_watch_config()
            target_tickers = config.get("tickers", [])
            target_keywords = config.get("keywords", [])
            
            # 2. Prepare text for keyword searching (lowercase everything)
            full_text = (headline + " " + summary).lower()
            
            matched = False
            trigger_reason = ""
            matched_ticker = ""

            # 3. Check Condition A: Is it a stock ticker we care about?
            for t in target_tickers:
                if t in symbols:
                    matched = True
                    trigger_reason = f"Ticker Match: {t}"
                    matched_ticker = t
                    break
            
            # 4. Check Condition B: Does it contain our macro geopolitical keywords?
            if not matched:
                for k in target_keywords:
                    if k.lower() in full_text:
                        matched = True
                        trigger_reason = f"Keyword Match: '{k}'"
                        matched_ticker = symbols[0] if symbols else "MACRO"
                        break

            # 5. If matched, print and trigger the AI!
            if matched:
                print(f"\n📰 [CAUGHT | {trigger_reason}] {headline}")
                threading.Thread(
                    target=trigger_graph_pipeline, 
                    args=(trigger_reason, matched_ticker, headline, summary),
                    daemon=True
                ).start()

def on_error(ws, error):
    print(f"⚠️ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔌 Alpaca WebSocket Connection Closed.")

def on_open(ws):
    print("🌐 Connecting to Alpaca News... Authenticating...")
    ws.send(json.dumps({
        "action": "auth",
        "key": settings.ALPACA_API_KEY,
        "secret": settings.ALPACA_SECRET_KEY
    }))

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "wss://stream.data.alpaca.markets/v1beta1/news",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    print("🚀 Alpha-Stream Keyword/Ticker Radar is online...")
    ws.run_forever()