"""
Pro-Investor — Streamlit UI
Dark financial theme, full RTL Hebrew interface.
All user-facing text is in Hebrew; code/comments in English.
"""

import json
import os
import socket
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))


# ── BackendClient: tries TCP server, falls back to direct calls ────────────────
class BackendClient:
    """
    Thin client that routes requests to the async TCP server (server.py) when
    reachable, and silently falls back to direct local function calls otherwise.
    The UI stays identical in both modes.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 0.5):
        self.host    = host
        self.port    = port
        self.timeout = timeout

    def _send(self, payload: dict) -> dict | None:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.sendall(json.dumps(payload).encode("utf-8"))
                sock.settimeout(15.0)
                chunks = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                resp = json.loads(b"".join(chunks).decode("utf-8").strip())
                return resp.get("data") if resp.get("success") else None
        except Exception:
            return None

    def score(self, ticker: str, risk: str, months: int) -> dict | None:
        return self._send({"action": "score", "ticker": ticker, "risk": risk, "months": months})

    def price(self, ticker: str) -> float | None:
        result = self._send({"action": "price", "ticker": ticker})
        return result.get("price") if result else None

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Pro-Investor | יועץ השקעות AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve GROQ_API_KEY: st.secrets first, then env ──────────────────────────
_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if _key:
    os.environ["GROQ_API_KEY"] = _key
else:
    st.error("### ⚠️ מפתח Groq API חסר")
    st.markdown(
        "הוסף לקובץ `.streamlit/secrets.toml`:\n```toml\nGROQ_API_KEY = \"gsk_...\"\n```"
    )
    st.stop()

from agent import InvestmentAgent
from data_engine import get_historical_data, get_multiple_prices
from scoring_engine import score_ticker
from database import init_db, get_or_create_user, save_recommendation, get_recent_recommendations, get_stats

# Ensure DB tables exist (safe to call on every run)
try:
    init_db()
except Exception:
    pass

# ── Dark Financial Theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== Base: force dark background ===== */
.stApp, .stApp > div {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
}

/* ===== RTL for all text ===== */
.stApp { direction: rtl; }
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stTextInput label {
    direction: rtl !important;
    text-align: right !important;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-left: 1px solid #1e293b !important;
}
section[data-testid="stSidebar"] * {
    direction: rtl;
    text-align: right;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #10b981 !important;
    color: white !important;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ===== Inputs ===== */
.stTextInput input, .stNumberInput input, .stChatInput textarea {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    direction: rtl;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.2) !important;
}
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ===== Chat messages ===== */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 4px 8px !important;
    margin-bottom: 6px !important;
    border: 1px solid #1e293b !important;
}
/* User message — right-aligned blue bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #1e3a5f33, #2d4a7a22) !important;
    border-color: #2d4a7a !important;
    margin-left: 12% !important;
}
/* Assistant message — left-aligned emerald bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, #06221544, #0a2e1e33) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
    margin-right: 12% !important;
}

/* ===== Metrics ===== */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 18px !important;
}
[data-testid="stMetricValue"] { color: #10b981 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* ===== Expander (tool log) ===== */
.streamlit-expanderHeader {
    background: #111827 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    direction: rtl;
}
.streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 0 0 10px 10px !important;
}

/* ===== DataTable ===== */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #1e293b !important;
}

/* ===== Divider ===== */
hr { border-color: #1e293b !important; }

/* ===== Spinner ===== */
.stSpinner > div { border-top-color: #10b981 !important; }

/* ===== Slider ===== */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #10b981 !important;
}

/* ===== Success / Error boxes ===== */
.stAlert { border-radius: 12px !important; direction: rtl; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "agent"        not in st.session_state:
    st.session_state.agent = InvestmentAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "backend"      not in st.session_state:
    st.session_state.backend = BackendClient()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Pro-Investor")
    st.markdown("*יועץ השקעות AI אוטונומי*")
    st.divider()

    st.markdown("### ⚙️ פרופיל המשקיע")

    budget = st.number_input(
        "💰 תקציב ($)",
        min_value=100, max_value=10_000_000, value=10_000, step=500, format="%d",
    )

    risk_map   = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
    risk_label = st.selectbox("⚖️ רמת סיכון", list(risk_map.keys()), index=1)
    risk       = risk_map[risk_label]

    duration = st.slider("📅 אופק (חודשים)", 1, 60, 12)

    st.divider()

    # ── Live mini-ticker bar ───────────────────────────────────────────────────
    st.markdown("### 📡 שוק בזמן אמת")
    WATCHLIST = ["SPY", "AAPL", "NVDA", "BTC-USD"]
    with st.spinner(""):
        try:
            prices = get_multiple_prices(WATCHLIST)
            for sym, price in prices.items():
                if price:
                    st.metric(sym, f"${price:,.2f}")
        except Exception:
            st.caption("נתוני שוק זמינים בצ'אט")

    st.divider()
    st.caption(f"🤖 LLM: LLaMA-3.3-70b via Groq")
    st.caption(f"🗄️ RAG: ChromaDB + Sentence-Transformers")
    st.caption(f"🔍 Web: DuckDuckGo Search")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:right; direction:rtl; color:#10b981;'>📈 Pro-Investor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:right; direction:rtl; color:#94a3b8;'>"
    "סוכן השקעות AI אוטונומי — מחפש ברשת, מנתח, ומדרג בזמן אמת</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_analyze, tab_chart, tab_history = st.tabs([
    "💬 ייעוץ AI", "📊 ניתוח שוק", "📈 גרפים", "📋 היסטוריה"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="📈"):
                st.markdown(msg["content"])
                # Show tool log as collapsible research trail
                log = msg.get("tool_log", [])
                if log:
                    with st.expander(f"🔍 תהליך מחקר הסוכן ({len(log)} צעדים)", expanded=False):
                        for step in log:
                            st.markdown(
                                f"{step['icon']} **{step['label']}** — {step['summary']}"
                            )

    # Chat input
    user_input = st.chat_input(
        "שאל את הסוכן... (לדוגמה: 'נתח NVDA עבור 12 חודשים' או 'מה המגמה ב-BTC?')"
    )

    if user_input:
        # Display user bubble immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        profile = {"budget": budget, "risk_tolerance": risk, "duration_months": duration}

        with st.chat_message("assistant", avatar="📈"):
            status = st.status("🔍 הסוכן חוקר...", expanded=True)
            status.write("מחפש נתונים בזמן אמת ומריץ ניתוח...")

            try:
                answer, tool_log = st.session_state.agent.run(user_input, profile)

                status.update(
                    label=f"✅ ניתוח הושלם — {len(tool_log)} כלים הופעלו",
                    state="complete",
                    expanded=False,
                )

                st.markdown(answer)

                if tool_log:
                    with st.expander(f"🔍 תהליך מחקר הסוכן ({len(tool_log)} צעדים)", expanded=False):
                        for step in tool_log:
                            st.markdown(
                                f"{step['icon']} **{step['label']}** — {step['summary']}"
                            )

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "tool_log": tool_log}
                )

                # Save recommendation to DB (best-effort — never blocks the UI)
                try:
                    profile = {"budget": budget, "risk_tolerance": risk, "duration_months": duration}
                    user_id = get_or_create_user(
                        budget=float(budget),
                        risk_tolerance=risk,
                        duration_months=int(duration),
                    )
                    for step in tool_log:
                        if step.get("tool") == "get_investment_score" and step.get("data"):
                            d = step["data"]
                            save_recommendation(
                                ticker=d.get("ticker", ""),
                                verdict=d.get("human", {}).get("verdict_raw", "WAIT"),
                                price=float(d.get("current_price", 0)),
                                score_data=d,
                                ai_response=answer,
                                user_id=user_id,
                            )
                            break
                except Exception:
                    pass

            except Exception as exc:
                status.update(label="❌ שגיאה", state="error")
                err = f"שגיאה: {exc}"
                st.error(err)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": err, "tool_log": []}
                )

    if st.session_state.chat_history:
        if st.button("🗑️ נקה שיחה"):
            st.session_state.chat_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Analysis (scoring table)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown(
        "<h3 style='text-align:right; direction:rtl;'>📊 ניתוח וציון ניירות ערך</h3>",
        unsafe_allow_html=True,
    )

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        tickers_raw = st.text_input(
            "טיקרים (מופרדים בפסיק):", value="AAPL, MSFT, NVDA, SPY, BTC-USD"
        )
    with col_btn:
        st.write("")
        run_analysis = st.button("🔍 נתח", key="analyze_btn")

    if run_analysis and tickers_raw:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        results = []
        bar = st.progress(0, text="מנתח...")

        for i, sym in enumerate(tickers):
            try:
                results.append(score_ticker(sym, risk, duration))
            except Exception as e:
                st.warning(f"⚠️ {sym}: {e}")
            bar.progress((i + 1) / len(tickers), text=f"מנתח {sym}...")

        bar.empty()

        if results:
            results.sort(key=lambda x: x["investment_score"], reverse=True)

            TREND = {"bullish": "🟢 עולה", "bearish": "🔴 יורד", "neutral": "🟡 ניטרלי"}

            df = pd.DataFrame([
                {
                    "#":             i + 1,
                    "טיקר":          r["ticker"],
                    "מחיר":          f"${r['current_price']:.2f}",
                    "יעד":           f"${r['target_price']:.2f}",
                    "תשואה צפויה":   f"{r['expected_return']*100:.1f}%",
                    "P(רווח)":       f"{r['probability_profit']*100:.0f}%",
                    "RSI":           f"{r['rsi']:.1f}",
                    "תנודתיות":      f"{r['volatility']*100:.1f}%",
                    "מגמה":          TREND.get(r["trend"], r["trend"]),
                    "ציון ⭐":        f"{r['investment_score']:.5f}",
                }
                for i, r in enumerate(results)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            best = results[0]
            st.success(
                f"### 🥇 המלצה מובילה: **{best['ticker']}**\n\n"
                f"ציון `{best['investment_score']:.5f}` | "
                f"תשואה `{best['expected_return']*100:.1f}%` | "
                f"{TREND.get(best['trend'], best['trend'])}"
            )

            with st.expander("📐 נוסחת הציון המתמטית"):
                st.latex(
                    r"Score = \frac{Probability_{Profit} \cdot Expected_{Return}}{Risk_{Factor}}"
                )
                st.markdown(f"""
| מרכיב | ערך |
|---|---|
| הסתברות רווח | `{best['probability_profit']}` |
| תשואה צפויה | `{best['expected_return']*100:.2f}%` |
| מקדם סיכון | `{best['risk_factor']}` |
| **ציון סופי** | **`{best['investment_score']:.5f}`** |
                """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Candlestick Chart
# ══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.markdown(
        "<h3 style='text-align:right; direction:rtl;'>📈 גרף נרות יפני</h3>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_sym = st.text_input("טיקר:", value="SPY", key="chart_sym")
    with c2:
        period_opts = {"שבוע": "5d", "חודש": "1mo", "3 חודשים": "3mo", "6 חודשים": "6mo", "שנה": "1y"}
        period_lbl  = st.selectbox("תקופה:", list(period_opts.keys()), index=2)

    if st.button("📊 הצג גרף", key="chart_btn"):
        with st.spinner("טוען נתונים..."):
            try:
                data = get_historical_data(chart_sym.upper(), period_opts[period_lbl])

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data["Open"], high=data["High"],
                    low=data["Low"],   close=data["Close"],
                    increasing_line_color="#10b981",
                    decreasing_line_color="#ef4444",
                    name=chart_sym.upper(),
                ))

                # 20-day MA overlay
                ma20 = data["Close"].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=ma20,
                    name="ממוצע נע 20",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ))

                fig.update_layout(
                    title=dict(text=f"{chart_sym.upper()} — {period_lbl}", x=0.5),
                    yaxis_title="מחיר ($)",
                    xaxis_title="תאריך",
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0d1117",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig, use_container_width=True)

                cur = float(data["Close"].iloc[-1])
                beg = float(data["Close"].iloc[0])
                pct = (cur - beg) / beg * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("מחיר נוכחי", f"${cur:.2f}")
                m2.metric("שינוי בתקופה", f"{pct:+.2f}%", delta=f"{pct:+.2f}%")
                m3.metric("שיא תקופה",   f"${data['High'].max():.2f}")
                m4.metric("שפל תקופה",   f"${data['Low'].min():.2f}")

            except Exception as exc:
                st.error(f"❌ שגיאה: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Recommendation History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown(
        "<h3 style='text-align:right; direction:rtl;'>📋 היסטוריית המלצות</h3>",
        unsafe_allow_html=True,
    )

    if st.button("🔄 רענן", key="refresh_history"):
        st.rerun()

    try:
        stats = get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("סה\"כ המלצות", stats["total"])
        c2.metric("✅ קנייה", stats["buys"])
        c3.metric("⚠️ המתן", stats["waits"])
        c4.metric("❌ הימנע", stats["avoids"])

        st.divider()

        recs = get_recent_recommendations(50)
        if recs:
            VERDICT_HE = {"BUY": "✅ קנייה", "WAIT": "⚠️ המתן", "AVOID": "❌ הימנע"}
            df_hist = pd.DataFrame([
                {
                    "תאריך":       r["date"],
                    "טיקר":        r["ticker"],
                    "פסיקה":       VERDICT_HE.get(r["verdict"], r["verdict"]),
                    "מחיר":        f"${r['price']:.2f}" if r["price"] else "—",
                    "ציון":        f"{r['score']:.4f}" if r["score"] else "—",
                    "תשואה צפויה": f"{r['return']*100:.1f}%" if r["return"] else "—",
                }
                for r in recs
            ])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("📭 אין המלצות שמורות עדיין. שאל את הסוכן על מניה בלשונית ייעוץ AI!")

    except Exception as exc:
        st.error(f"❌ שגיאת בסיס נתונים: {exc}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "⚠️ המידע הוא לצורכי מידע בלבד ואינו ייעוץ פיננסי מוסמך. "
    "השקעות כרוכות בסיכון אובדן הון. | Pro-Investor AI © 2025</p>",
    unsafe_allow_html=True,
)
