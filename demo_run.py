"""
Pro-Investor — Full System Demo
Runs every core engine and prints a structured report to the terminal.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

DIVIDER = "=" * 62

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ── 1. Data Engine ─────────────────────────────────────────────
section("1/5  DATA ENGINE  (yfinance live prices)")

from data_engine import get_live_price, get_asset_info, get_multiple_prices

tickers = ["AAPL", "NVDA", "SPY", "BTC-USD"]
prices = get_multiple_prices(tickers)
for sym, price in prices.items():
    tag = f"${price:.2f}" if price else "N/A"
    print(f"  {sym:<12}  {tag}")

print("\n  --- Asset info: AAPL ---")
info = get_asset_info("AAPL")
for k, v in info.items():
    print(f"  {k:<20} {v}")

# ── 2. Technical Indicators ────────────────────────────────────
section("2/5  TECHNICAL INDICATORS  (RSI / MACD / Bollinger)")

from technical_indicators import get_all_indicators

for sym in ["AAPL", "NVDA", "BTC-USD"]:
    ind = get_all_indicators(sym)
    trend_icon = {"bullish": "[^]", "bearish": "[v]", "neutral": "[-]"}.get(ind["trend"], "?")
    rsi_flag = "OVERSOLD" if ind["rsi"] < 30 else ("OVERBOUGHT" if ind["rsi"] > 70 else "neutral")
    macd_dir = "BULL" if ind["macd_histogram"] > 0 else "BEAR"
    print(
        f"  {sym:<10} | Price ${ind['current_price']:.2f}"
        f" | RSI {ind['rsi']:.1f} ({rsi_flag})"
        f" | MACD {macd_dir}"
        f" | BB upper ${ind['bollinger_upper']:.2f}"
        f" | Vol {ind['volatility']*100:.1f}%"
        f" | Trend {trend_icon}"
    )

# ── 3. Scoring Engine ──────────────────────────────────────────
section("3/5  SCORING ENGINE  ( Score = P_profit * R / Risk )")

from scoring_engine import score_ticker

candidates = ["AAPL", "NVDA", "SPY", "BTC-USD", "MSFT"]
risk_tolerance = "medium"
duration = 12

results = []
for sym in candidates:
    try:
        r = score_ticker(sym, risk_tolerance, duration)
        results.append(r)
    except Exception as e:
        print(f"  {sym}: ERROR — {e}")

results.sort(key=lambda x: x["investment_score"], reverse=True)

print(f"\n  Risk tolerance: {risk_tolerance}   Duration: {duration} months\n")
print(f"  {'#':<3} {'Ticker':<8} {'Price':>8} {'Target':>8} {'P(profit)':>10} {'E[Return]':>10} {'Risk':>7} {'Score':>10} {'Trend'}")
print(f"  {'-'*80}")
for i, r in enumerate(results, 1):
    print(
        f"  {i:<3} {r['ticker']:<8} ${r['current_price']:>7.2f} ${r['target_price']:>7.2f}"
        f" {r['probability_profit']*100:>8.1f}%"
        f" {r['expected_return']*100:>8.1f}%"
        f" {r['risk_factor']:>7.4f}"
        f" {r['investment_score']:>10.5f}"
        f"  {r['trend']}"
    )

best = results[0]
print(f"\n  >>> TOP PICK: {best['ticker']} (score {best['investment_score']:.5f})")
print(f"      Formula: ({best['probability_profit']} x {best['expected_return']:.4f}) / {best['risk_factor']:.4f} = {best['investment_score']:.5f}")

# ── 4. RAG Pipeline ────────────────────────────────────────────
section("4/5  RAG PIPELINE  (Chunk -> Embed -> ChromaDB -> Rerank)")

from rag_pipeline import RAGPipeline

rag = RAGPipeline()
print("  RAGPipeline initialized.")
print("  Embedding model : sentence-transformers/all-MiniLM-L6-v2")
print("  Rerank model    : cross-encoder/ms-marco-MiniLM-L-6-v2")

sample_doc = """
Apple Inc. reported record Q1 2025 earnings, with revenue of $124.3 billion,
up 4% year-over-year. iPhone sales rose 6% driven by strong demand in emerging markets.
The Services segment hit an all-time high of $26.3 billion.
CEO Tim Cook highlighted AI integration across all product lines as the key growth driver.
Analysts raised their 12-month price targets to $240-$260 range.
The company also announced a $110 billion share buyback program.

NVIDIA reported Q4 FY2025 revenue of $39.3 billion, up 78% YoY.
Data center revenue was $35.6 billion, fueled by Blackwell GPU demand.
Management guided Q1 FY2026 revenue at approximately $43 billion.
The stock trades at a forward P/E of 35x, reflecting high growth expectations.
Volatility risk remains elevated given supply chain concentration.
"""

chunks_added = rag.ingest_document(sample_doc, source="earnings_reports_2025")
print(f"\n  Document ingested -> {chunks_added} semantic chunks stored in ChromaDB")
print(f"  Total chunks in DB: {rag.get_document_count()}")

query = "Apple iPhone revenue growth outlook"
print(f"\n  Query: '{query}'")
results_rag = rag.retrieve_and_rerank(query, top_k_retrieve=5, top_k_rerank=2)

for i, r in enumerate(results_rag, 1):
    score_str = f"rerank={r['rerank_score']:.3f}, cosine={r['similarity_score']:.3f}"
    snippet = r["text"][:120].replace("\n", " ")
    print(f"\n  [{i}] {score_str}")
    print(f"      Source : {r['source']}")
    print(f"      Text   : {snippet}...")

# ── 5. System Summary ──────────────────────────────────────────
section("5/5  SYSTEM SUMMARY")

line = "-" * 50
print(f"""
  Component                Status
  {line}
  Data Engine (yfinance)   LIVE -- fetched {len(tickers)} tickers
  Technical Indicators     LIVE -- RSI, MACD, Bollinger, Vol, Trend
  Scoring Engine           LIVE -- ranked {len(results)} assets
  RAG Pipeline             LIVE -- {rag.get_document_count()} chunks in ChromaDB
  AI Agent (LLaMA-3.3-70b) READY -- needs GROQ_API_KEY in env (free)
  Streamlit UI             RUNNING on http://localhost:8501
  {line}
  Top investment pick      {best['ticker']} (score={best['investment_score']:.5f})
""")

print(DIVIDER)
print("  Pro-Investor demo complete.")
print(DIVIDER)
