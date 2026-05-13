"""
Autonomous AI Investment Agent — powered by Groq (free LLM API).

Key features:
  - Pydantic input models: strict schema enforcement + automatic type coercion
  - Friendly-guide persona: warm Hebrew responses, graceful error recovery in Hebrew
  - web_search_and_learn: DuckDuckGo search → auto-stores results in ChromaDB
  - get_investment_score: returns only human-readable Hebrew labels (no raw decimals)
  - run() returns (answer: str, tool_log: list) for UI display
"""

import json
import os
from datetime import datetime

from groq import Groq

from data_engine import get_live_price, get_asset_info
from technical_indicators import get_all_indicators
from scoring_engine import score_ticker, score_to_human_readable
from rag_pipeline import RAGPipeline

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Pydantic validation models — coerce and validate LLM tool arguments ────────
# Pydantic 1.x is installed as a transitive dep of chromadb==0.4.24.
# These models guarantee correct types even when the LLM sends "12" (str) for
# duration_months (int), preventing Groq 400 "expected integer, got string" errors.
try:
    from pydantic import BaseModel
    from pydantic import validator as _v

    class _ScoreInput(BaseModel):
        ticker_symbol: str = "AAPL"
        risk_tolerance: str = "medium"
        duration_months: int = 12

        @_v("ticker_symbol", pre=True, always=True)
        @classmethod
        def _t(cls, v):
            return str(v).strip().upper() or "AAPL"

        @_v("risk_tolerance", pre=True, always=True)
        @classmethod
        def _r(cls, v):
            s = str(v).lower().strip()
            return s if s in ("low", "medium", "high") else "medium"

        @_v("duration_months", pre=True, always=True)
        @classmethod
        def _m(cls, v):
            return max(1, min(120, int(float(str(v)))))

    class _TickerInput(BaseModel):
        ticker_symbol: str

        @_v("ticker_symbol", pre=True, always=True)
        @classmethod
        def _t(cls, v):
            return str(v).strip().upper()

    _PYDANTIC = True

except Exception:
    _PYDANTIC = False


def _parse_score(raw: dict, risk_default: str, months_default: int) -> tuple:
    """Returns (ticker: str, risk: str, months: int) with guaranteed correct types."""
    merged = {"risk_tolerance": risk_default, "duration_months": months_default, **raw}
    if _PYDANTIC:
        try:
            m = _ScoreInput(**merged)
            return m.ticker_symbol, m.risk_tolerance, m.duration_months
        except Exception:
            pass
    # Manual fallback when Pydantic unavailable
    sym = str(raw.get("ticker_symbol", "")).strip().upper()
    rt  = str(raw.get("risk_tolerance", risk_default)).lower().strip()
    rt  = rt if rt in ("low", "medium", "high") else "medium"
    dm  = max(1, int(float(str(raw.get("duration_months", months_default)))))
    return sym, rt, dm


def _parse_ticker(raw: dict) -> str:
    if _PYDANTIC:
        try:
            return _TickerInput(**raw).ticker_symbol
        except Exception:
            pass
    return str(raw.get("ticker_symbol", "")).strip().upper()


# ── Hebrew ticker map & intent classification ──────────────────────────────────

HEBREW_TICKER_MAP: dict[str, str] = {
    # US Tech
    "אפל": "AAPL", "apple": "AAPL",
    "גוגל": "GOOGL", "אלפבית": "GOOGL", "alphabet": "GOOGL",
    "אמזון": "AMZN", "amazon": "AMZN",
    "מטא": "META", "פייסבוק": "META", "facebook": "META",
    "נבידיה": "NVDA", "אנבידיה": "NVDA", "nvidia": "NVDA", "נבידייה": "NVDA",
    "טסלה": "TSLA", "תיסלה": "TSLA", "tesla": "TSLA",
    "מיקרוסופט": "MSFT", "מייקרוסופט": "MSFT", "microsoft": "MSFT",
    "אינטל": "INTC", "intel": "INTC",
    "נטפליקס": "NFLX", "netflix": "NFLX",
    "ספוטיפיי": "SPOT", "spotify": "SPOT",
    "אובר": "UBER", "uber": "UBER",
    "פייפאל": "PYPL", "paypal": "PYPL",
    "אדובי": "ADBE", "adobe": "ADBE",
    "סיילספורס": "CRM", "salesforce": "CRM",
    "קוואלקום": "QCOM", "qualcomm": "QCOM",
    "AMD": "AMD", "אמדי": "AMD",
    # Indices
    "נסדאק": "^IXIC", "נסדק": "^IXIC", "nasdaq": "^IXIC",
    "ספ500": "^GSPC", "sp500": "^GSPC", "s&p": "^GSPC", "s&p500": "^GSPC",
    "דאו": "^DJI", "דאו ג'ונס": "^DJI", "dow": "^DJI",
    "ראסל": "^RUT", "russell": "^RUT",
    # ETFs
    "spy": "SPY", "qqq": "QQQ", "vix": "^VIX",
    # Crypto
    "ביטקוין": "BTC-USD", "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "איתריום": "ETH-USD", "את'ריום": "ETH-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
    "סולנה": "SOL-USD", "solana": "SOL-USD",
    "ריפל": "XRP-USD", "ripple": "XRP-USD",
    # Commodities
    "זהב": "GC=F", "gold": "GC=F",
    "נפט": "CL=F", "oil": "CL=F", "crude": "CL=F",
    "כסף": "SI=F", "silver": "SI=F",
    # Finance
    "ג'יי פי מורגן": "JPM", "jp morgan": "JPM", "jpmorgan": "JPM",
    "בנק אוף אמריקה": "BAC", "bank of america": "BAC",
    "גולדמן": "GS", "גולדמן זאקס": "GS", "goldman": "GS",
    "ברקשייר": "BRK-B", "ברקשייר הת'אווי": "BRK-B",
    # Consumer
    "קוקה קולה": "KO", "קוקה": "KO", "coca cola": "KO",
    "וולמארט": "WMT", "walmart": "WMT",
    "דיסני": "DIS", "disney": "DIS",
    # Healthcare
    "פייזר": "PFE", "pfizer": "PFE",
    "ג'ונסון": "JNJ", "johnson": "JNJ",
    # Energy
    "אקסון": "XOM", "exxon": "XOM",
    "שברון": "CVX", "chevron": "CVX",
    # Israeli
    "לאומי": "LUMI.TA", "בנק לאומי": "LUMI.TA",
    "פועלים": "POLI.TA", "בנק פועלים": "POLI.TA",
    "ת\"א 35": "^TA35", "ta35": "^TA35", "תל אביב 35": "^TA35",
    # FX
    "דולר שקל": "ILS=X", "שקל": "ILS=X",
    "יורו": "EURUSD=X", "euro": "EURUSD=X",
}

# Keywords that signal a financial query
_FINANCIAL_SIGNALS: frozenset[str] = frozenset([
    "מניה", "מניות", "שוק", "בורסה", "השקעה", "השקעות", "תיק", "קריפטו",
    "ביטקוין", "נסדאק", "נסדק", "sp500", "ספ500", "דאו", "תשואה",
    "דיבידנד", "etf", "ipo", "רווח", "הפסד", "אג\"ח", "קנייה", "מכירה",
    "אפל", "גוגל", "אמזון", "מטא", "נבידיה", "טסלה", "מיקרוסופט",
    "stock", "share", "invest", "portfolio", "crypto", "bitcoin", "market",
    "nasdaq", "s&p", "bond", "equity", "trade", "earnings", "dividend",
    "price", "rsi", "macd", "ticker", "fund", "זהב", "נפט", "ניתוח",
    "לנתח", "לבדוק", "שווי", "מחיר", "כדאי", "לקנות", "למכור",
])

# Keywords that signal a clearly off-topic query
_OFF_TOPIC_SIGNALS: frozenset[str] = frozenset([
    "לברון", "לבראון", "קובי", "ברייאנט", "כדורסל", "כדורגל",
    "מסי", "רונאלדו", "ספורט", "נבחרת", "ליגה", "גביע", "nba", "nfl",
    "fifa", "champions", "שחקן כדור", "טורניר",
    "נשיא ארה\"ב", "ראש ממשלה", "בחירות", "מפלגה",
    "סרט", "אלבום", "זמר", "שחקן קולנוע", "טלוויזיה", "שיר",
    "מזג אוויר", "בירה של", "בירת",
])


def _classify_intent(query: str) -> str:
    """
    Fast keyword pre-classifier.
    Returns 'financial', 'off_topic', or 'ambiguous' (let LLM decide).
    """
    q = query.lower()
    has_fin     = any(kw in q for kw in _FINANCIAL_SIGNALS)
    has_offtop  = any(kw in q for kw in _OFF_TOPIC_SIGNALS)

    if has_fin and not has_offtop:
        return "financial"
    if has_offtop and not has_fin:
        return "off_topic"
    return "ambiguous"   # LLM decides via system-prompt instruction


def _fuzzy_resolve_ticker(name: str) -> str | None:
    """
    Translates a Hebrew company name / common alias to a ticker.
    Exact match → substring match → None (let LLM reason it out).
    """
    n = name.lower().strip()
    if n in HEBREW_TICKER_MAP:
        return HEBREW_TICKER_MAP[n]
    for key, ticker in HEBREW_TICKER_MAP.items():
        if key in n or n in key:
            return ticker
    return None


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Pro-Investor — a warm, encouraging financial guide who speaks to everyday investors in simple, friendly Hebrew. You are NOT a Wall Street analyst. You are a knowledgeable friend who makes investing approachable and understandable.

PERSONALITY:
  - Always respond 100% in Hebrew — simple, clear language, zero financial jargon
  - Be encouraging and honest: celebrate opportunities, be transparent about risks
  - If a tool fails: "אני מצטער, הייתה לי בעיה טכנית במשיכת הנתונים עבור [הנכס]. אנסה גישה אחרת."
    Then retry immediately or provide a partial answer from what you know
  - Never leave the user with an unexplained technical error

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — INTENT CLASSIFICATION (before ANY tool call):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the query is NOT about finance/stocks/crypto/markets → respond immediately with NO tools:
  "אני מומחה להשקעות ופיננסים, ולכן פחות מבין ב[נושא]. אשמח לעזור לך לנתח מניות או לבנות תיק השקעות!"

Off-topic examples (no tools, polite decline):
  • ספורט: "מי טוב יותר לברון או קובי?"
  • פוליטיקה: "מי ינצח בבחירות?"
  • ידע כללי: "מה הבירה של צרפת?"

Financial examples (proceed to tools):
  • "מה שווה אפל כרגע?" / "כדאי לקנות נסדק?" / "ביטקוין — קנייה או מכירה?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — TICKER RESOLUTION (before calling any tool):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always resolve Hebrew names to tickers BEFORE tool calls:
  אפל→AAPL | גוגל/אלפבית→GOOGL | אמזון→AMZN | מטא/פייסבוק→META
  נבידיה/אנבידיה/נבידייה→NVDA | טסלה/תיסלה→TSLA | מיקרוסופט→MSFT
  נסדאק/נסדק→^IXIC | ספ500→^GSPC | דאו→^DJI | ת"א 35→^TA35
  ביטקוין→BTC-USD | איתריום→ETH-USD | זהב→GC=F | נפט→CL=F
  לאומי→LUMI.TA | פועלים→POLI.TA

For typos (נסדק, נבידייה, תיסלה, etc.) — use your reasoning to infer the correct ticker.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — TOOL SELECTION (call only what the query actually needs):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match tool depth to query complexity:

  מחיר בלבד ("כמה שווה AAPL?"):
    → get_live_price ONLY

  מידע חברה ("ספר לי על אמזון"):
    → get_asset_info ONLY

  מגמה/מומנטום ("מה המגמה של NVDA?"):
    → get_live_price + get_technical_indicators

  המלצת השקעה מלאה ("כדאי לקנות?", "תנתח את"):
    → web_search_and_learn → get_technical_indicators → get_investment_score → search_knowledge_base

CHAIN OF THOUGHT — before calling any tool, think one line:
  "המשתמש שאל [X]. אני צריך [כלים Y] בלבד כי [סיבה]."
  Then call ONLY those tools. Never call extra tools "just in case."
  Hard limit: maximum 4 tool calls per response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT (full stock recommendation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**שם הנכס:** [Company Name — TICKER]

**למה כדאי?** (בשפה פשוטה):
[One clear sentence — explain the opportunity like you would to a friend]

**מה הפוטנציאל?**
[Use score_label + return_str from get_investment_score]

**מגמה ומומנטום:** [Use trend_he from get_investment_score]

**רמת סיכון:** [Use risk_label from get_investment_score]

**שורה תחתונה:** [verdict field: ✅ קנייה / ⚠️ המתן / ❌ לא כרגע]

**רמת ודאות הסוכן:**
  גבוהה 🟢 — web_search + technical + score כולם החזירו נתונים תקינים
  בינונית 🟡 — 2 מתוך 3 מקורות עבדו
  נמוכה 🔴 — מקור אחד בלבד, או היו שגיאות טכניות

---
📰 **מקורות:**
[Format EACH URL from web_search_and_learn as a clickable markdown link: [כותרת המאמר](url)]

💡 **שלב הבא:** [Suggest ONE specific follow-up action, e.g., "תרצה שאשווה את AAPL מול MSFT?" or "אבדוק גם את ביצועי הסקטור?"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - NEVER show raw decimals (0.55, 0.127) — Hebrew labels only
  - NEVER call more than 4 tools per response
  - duration_months is auto-coerced to integer — just pass a number
  - risk_tolerance is auto-validated — just pass low/medium/high
  - Always greet the user by name on the first message
  - Always end full recommendations with 📰 מקורות + 💡 שלב הבא"""


# ── Tool schemas ───────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search_and_learn",
            "description": (
                "Search the web for current financial news, analyst reports, or market data. "
                "Results are automatically stored in the local knowledge base for future queries. "
                "Use this FIRST for any question about a specific stock or market event."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query, e.g. 'Apple AAPL stock earnings 2025'",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Related ticker symbol if applicable, e.g. 'AAPL'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_price",
            "description": "Fetches the current real-time market price of any financial asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Ticker, e.g. 'AAPL', 'BTC-USD', 'SPY', 'NVDA'",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Computes RSI, MACD, Bollinger Bands, annualized volatility and trend from 6 months of price history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Ticker to analyze",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_investment_score",
            "description": (
                "Calculates Score = (Probability_Profit × Expected_Return) / Risk_Factor. "
                "Returns ONLY human-readable Hebrew labels — no raw decimals exposed. "
                "All inputs are automatically validated and type-coerced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Ticker to score",
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "User risk level: low / medium / high",
                    },
                    "duration_months": {
                        "type": "integer",
                        "description": "Investment horizon as a whole number of months, e.g. 12",
                    },
                },
                "required": ["ticker_symbol", "risk_tolerance", "duration_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Semantic search (cosine similarity + reranking) in local ChromaDB for previously stored news and analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_asset_info",
            "description": "Fundamental data: company name, sector, market cap, P/E, forward P/E, 52-week high/low, dividend yield.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Ticker to look up",
                    }
                },
                "required": ["ticker_symbol"],
            },
        },
    },
]

# Icon + Hebrew label per tool (for UI display)
TOOL_META = {
    "web_search_and_learn":    ("🌐", "חיפוש אינטרנט"),
    "get_live_price":          ("💰", "מחיר חי"),
    "get_technical_indicators":("📊", "אינדיקטורים טכניים"),
    "get_investment_score":    ("🎯", "ציון השקעה"),
    "search_knowledge_base":   ("🗄️", "מאגר ידע"),
    "get_asset_info":          ("🏢", "מידע פונדמנטלי"),
}


class InvestmentAgent:
    """
    Autonomous investment agent: Groq LLM + local tools + DuckDuckGo web search.
    Pydantic models coerce LLM tool inputs to correct types before execution.
    On tool failure, returns a Hebrew error so the LLM can recover gracefully.
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.rag = RAGPipeline()

    def _ddg_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Performs a DuckDuckGo text search; returns list of result dicts."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            return [{"title": "Search error", "body": str(exc), "href": ""}]

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        risk_tolerance: str,
        duration_months: int,
    ) -> tuple[str, dict]:
        """
        Runs one tool call. Pydantic models coerce inputs before execution.
        Returns (result_string_for_llm, parsed_dict_for_ui).
        On any error: returns a Hebrew error message so the LLM can explain
        the issue to the user and try an alternative approach.
        """
        parsed: dict = {}
        try:
            # ── Web search + auto-store in ChromaDB ───────────────────────────
            if tool_name == "web_search_and_learn":
                ticker = str(tool_input.get("ticker", "")).strip()
                query  = str(tool_input.get("query", "")).strip()
                full_q = f"{ticker} {query}".strip() if ticker else query

                results = self._ddg_search(full_q)
                if not results or not results[0].get("body"):
                    return "לא נמצאו תוצאות חיפוש.", {"count": 0}

                doc_parts = [
                    f"Title: {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                    for r in results
                ]
                combined = "\n\n".join(doc_parts)
                tag = f"web_{ticker or 'search'}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.rag.ingest_document(combined, source=tag)

                parsed = {
                    "count": len(results),
                    "stored": True,
                    "sources": [
                        {"title": r.get("title", ""), "url": r.get("href", "")}
                        for r in results if r.get("href")
                    ],
                }
                formatted = "\n\n---\n\n".join(
                    f"**{r.get('title','')}**\n{r.get('body','')}\n"
                    f"מקור: [{r.get('title', r.get('href',''))}]({r.get('href','')})"
                    for r in results
                )
                return formatted, parsed

            # ── Live price ────────────────────────────────────────────────────
            elif tool_name == "get_live_price":
                sym   = _parse_ticker(tool_input)
                price = get_live_price(sym)
                parsed = {"ticker": sym, "price": round(price, 2)}
                return json.dumps(parsed), parsed

            # ── Technical indicators ──────────────────────────────────────────
            elif tool_name == "get_technical_indicators":
                sym    = _parse_ticker(tool_input)
                result = get_all_indicators(sym)
                parsed = result
                return json.dumps(result), parsed

            # ── Investment score (Pydantic-coerced inputs) ────────────────────
            elif tool_name == "get_investment_score":
                sym, rt, dm = _parse_score(tool_input, risk_tolerance, duration_months)

                result = score_ticker(sym, rt, dm)
                result.pop("indicators", None)
                human  = score_to_human_readable({**result, "ticker": sym, "duration_months": dm})
                result["human"] = human
                parsed = result

                # Return only human-readable labels to LLM — never raw decimals
                llm_view = {
                    "ticker":        sym,
                    "current_price": f"${result['current_price']:.2f}",
                    "target_price":  f"${result['target_price']:.2f}",
                    "score_label":   human["score_label"],
                    "risk_label":    human["risk_label"],
                    "potential":     human["potential"],
                    "return_str":    human["return_str"],
                    "verdict":       human["verdict"],
                    "verdict_raw":   human["verdict_raw"],
                    "trend_he":      human["trend_he"],
                    "summary_line":  human["summary_line"],
                }
                return json.dumps(llm_view, ensure_ascii=False), parsed

            # ── RAG search ────────────────────────────────────────────────────
            elif tool_name == "search_knowledge_base":
                context = self.rag.get_context_string(str(tool_input.get("query", "")))
                parsed  = {"found": len(context) > 60}
                return context, parsed

            # ── Asset fundamentals ────────────────────────────────────────────
            elif tool_name == "get_asset_info":
                sym  = _parse_ticker(tool_input)
                info = get_asset_info(sym)
                parsed = info
                return json.dumps(info), parsed

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"}), {}

        except Exception as exc:
            # Hebrew error returned to LLM so it can explain the issue gracefully
            err_he = f"הכלי '{tool_name}' נתקל בשגיאה טכנית: {exc}"
            return err_he, {"error": str(exc), "tool": tool_name}

    def _build_tool_log_entry(self, tool_name: str, tool_input: dict, parsed: dict) -> dict:
        """Creates a structured log entry for UI rendering. Includes raw parsed data for DB logging."""
        icon, label = TOOL_META.get(tool_name, ("🔧", tool_name))

        if tool_name == "web_search_and_learn":
            count   = parsed.get("count", 0)
            sources = parsed.get("sources", [])
            src_str = " | ".join(s["title"][:30] for s in sources[:3] if s.get("title"))
            summary = f"חיפוש: \"{tool_input.get('query', '')}\" — {count} תוצאות" + (f" ({src_str})" if src_str else "")
        elif tool_name == "get_live_price":
            summary = f"{parsed.get('ticker', '')} — ${parsed.get('price', '?')}"
        elif tool_name == "get_technical_indicators":
            rsi   = parsed.get("rsi", "?")
            trend = parsed.get("trend", "?")
            macd  = "חיובי" if parsed.get("macd_histogram", 0) > 0 else "שלילי"
            summary = f"RSI={rsi:.1f}, מגמה={trend}, MACD={macd}" if isinstance(rsi, float) else f"מגמה={trend}"
        elif tool_name == "get_investment_score":
            human   = parsed.get("human", {})
            verdict = human.get("verdict", "?")
            ret_str = human.get("return_str", "")
            summary = f"{verdict} | {ret_str}" if ret_str else verdict
        elif tool_name == "search_knowledge_base":
            summary = "נמצא מידע רלוונטי במאגר" if parsed.get("found") else "לא נמצא מידע קודם"
        elif tool_name == "get_asset_info":
            mc = parsed.get("market_cap", 0)
            summary = f"{parsed.get('name', '')} | שווי שוק: ${mc/1e9:.1f}B" if mc else parsed.get("name", "")
        else:
            summary = str(parsed)[:80]

        return {"icon": icon, "label": label, "summary": summary, "tool": tool_name, "data": parsed}

    def run(self, user_query: str, user_profile: dict) -> tuple[str, list]:
        """
        Runs the agentic loop. Returns:
          - answer (str): final Hebrew recommendation
          - tool_log (list[dict]): each tool call, for UI display
        """
        risk_tolerance  = str(user_profile.get("risk_tolerance", "medium")).lower()
        duration_months = max(1, int(float(str(user_profile.get("duration_months", 6)))))
        budget          = float(user_profile.get("budget", 0))
        user_name       = str(user_profile.get("user_name", "")).strip()

        name_line = f"שם המשתמש: {user_name}. בשיחה הראשונה ברך אותו בשמו בחמימות.\n" if user_name else ""
        system_msg = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{name_line}"
            f"פרופיל משקיע: תקציב=${budget:,.0f} | סיכון={risk_tolerance} | אופק={duration_months} חודשים"
        )

        # ── Pre-flight: block clearly off-topic queries without an LLM call ─────
        intent = _classify_intent(user_query)
        if intent == "off_topic":
            # Extract the topic word for a personalised decline
            decline = (
                "אני מומחה להשקעות ופיננסים, ולכן פחות מבין בנושא הזה. "
                "אשמח לעזור לך לנתח מניות, לבדוק מחירים, או לבנות תיק השקעות! 📈"
            )
            return decline, []

        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_query},
        ]
        tool_log: list[dict] = []

        # Agentic loop — runs until finish_reason == "stop"
        while True:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                tc_list = choice.message.tool_calls

                messages.append({
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id":   tc.id,
                            "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tc_list
                    ],
                })

                for tc in tc_list:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}

                    result_str, parsed = self._execute_tool(
                        tc.function.name, tool_input, risk_tolerance, duration_months
                    )

                    tool_log.append(
                        self._build_tool_log_entry(tc.function.name, tool_input, parsed)
                    )

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         tc.function.name,
                        "content":      result_str,
                    })

            else:
                # finish_reason == "stop" → final answer
                return choice.message.content or "", tool_log
