"""
Pro-Investor — Streamlit UI
All labels, titles, and user-facing text are in Hebrew (RTL).
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))

# ── Resolve GROQ_API_KEY: prefer st.secrets, fall back to os.environ ──────────
_groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key  # make available to agent.py / groq SDK
else:
    st.set_page_config(page_title="Pro-Investor | הגדרה", page_icon="📈")
    st.error("### ⚠️ מפתח Groq API חסר")
    st.markdown(
        """
**לא נמצא `GROQ_API_KEY`** — הוסף אותו לקובץ `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_..."
```

קבל מפתח חינמי בכתובת: **https://console.groq.com/keys**
        """
    )
    st.stop()

from agent import InvestmentAgent
from data_engine import get_historical_data
from rag_pipeline import RAGPipeline
from scoring_engine import score_ticker

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pro-Investor | יועץ השקעות AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── RTL + custom styles ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    html, body, [class*="css"] { direction: rtl; }
    .stChatMessage { direction: rtl; text-align: right; }
    .main .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] { direction: rtl; }
    .formula-box {
        background: #0e1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-family: monospace;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = InvestmentAgent()
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar — investor profile ─────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ פרופיל המשקיע")
    st.divider()

    budget = st.number_input(
        "💰 תקציב השקעה ($)",
        min_value=100,
        max_value=10_000_000,
        value=10_000,
        step=500,
        format="%d",
    )

    risk_label_map = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
    risk_label = st.selectbox("⚖️ רמת סיכון", list(risk_label_map.keys()), index=1)
    risk_tolerance = risk_label_map[risk_label]

    duration = st.slider("📅 אופק השקעה (חודשים)", min_value=1, max_value=60, value=12, step=1)

    st.divider()
    st.subheader("📚 מאגר ידע RAG")

    knowledge_text = st.text_area(
        "הכנס דוח/חדשות פיננסיות:",
        height=140,
        placeholder="הדבק כאן טקסט של דוח כספי, ניתוח שוק, חדשות השקעות...",
    )
    knowledge_source = st.text_input("שם המקור:", value="financial_report")

    if st.button("📥 שמור במאגר הידע", use_container_width=True):
        if knowledge_text.strip():
            count = st.session_state.rag.ingest_document(knowledge_text, knowledge_source)
            total = st.session_state.rag.get_document_count()
            st.success(f"✅ נוספו {count} קטעים. סה״כ במאגר: {total}")
        else:
            st.warning("⚠️ יש להכניס טקסט לפני השמירה.")

    total_docs = st.session_state.rag.get_document_count()
    st.info(f"📊 קטעים במאגר: {total_docs}")

# ── Main header ────────────────────────────────────────────────────────────────
st.title("📈 Pro-Investor | יועץ השקעות AI")
st.caption(
    "מערכת המלצות השקעה חכמה המשלבת בינה מלאכותית, ניתוח טכני, RAG ונתוני שוק בזמן אמת"
)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_analyze, tab_chart = st.tabs(["💬 ייעוץ AI", "📊 ניתוח ניירות ערך", "📈 גרף שוק"])

# ── Tab 1: AI Chat ─────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("💬 שוחח עם יועץ ה-AI שלך")

    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        avatar = "👤" if role == "user" else "📈"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "שאל את יועץ ה-AI... (לדוגמה: 'נתח את NVDA לתקציב שלי' או 'מה ה-RSI של AAPL?')"
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        user_profile = {
            "budget": budget,
            "risk_tolerance": risk_tolerance,
            "duration_months": duration,
        }

        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("🤔 מנתח נתוני שוק בזמן אמת..."):
                try:
                    response = st.session_state.agent.run(user_input, user_profile)
                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as exc:
                    err_msg = f"❌ שגיאה: {exc}"
                    st.error(err_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err_msg}
                    )

    if st.session_state.chat_history:
        if st.button("🗑️ נקה שיחה"):
            st.session_state.chat_history = []
            st.rerun()

# ── Tab 2: Security Analysis ───────────────────────────────────────────────────
with tab_analyze:
    st.subheader("📊 ניתוח וציון ניירות ערך")
    st.markdown(
        "הזן רשימת טיקרים וקבל ניתוח טכני מלא עם ציון השקעה מתמטי עבור הפרופיל שלך."
    )

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        tickers_input = st.text_input(
            "טיקרים לניתוח (מופרדים בפסיק):",
            value="AAPL, MSFT, NVDA, SPY, BTC-USD",
        )
    with col_btn:
        st.write("")
        analyze_btn = st.button("🔍 נתח", use_container_width=True, type="primary")

    if analyze_btn and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        results = []
        progress_bar = st.progress(0, text="מנתח נתונים...")

        for idx, ticker in enumerate(tickers):
            try:
                data = score_ticker(ticker, risk_tolerance, duration)
                results.append(data)
            except Exception as exc:
                st.warning(f"⚠️ {ticker}: {exc}")
            progress_bar.progress((idx + 1) / len(tickers), text=f"מנתח {ticker}...")

        progress_bar.empty()

        if results:
            results.sort(key=lambda x: x["investment_score"], reverse=True)

            st.subheader("🏆 דירוג ניירות ערך")

            trend_emoji = {"bullish": "🟢 עולה", "bearish": "🔴 יורד", "neutral": "🟡 ניטרלי"}

            df = pd.DataFrame(
                [
                    {
                        "דירוג": i + 1,
                        "טיקר": r["ticker"],
                        "מחיר נוכחי": f"${r['current_price']:.2f}",
                        "מחיר יעד": f"${r['target_price']:.2f}",
                        "תשואה צפויה": f"{r['expected_return'] * 100:.1f}%",
                        "הסתברות רווח": f"{r['probability_profit'] * 100:.0f}%",
                        "RSI": f"{r['rsi']:.1f}",
                        "תנודתיות": f"{r['volatility'] * 100:.1f}%",
                        "מגמה": trend_emoji.get(r["trend"], r["trend"]),
                        "ציון השקעה ⭐": f"{r['investment_score']:.5f}",
                    }
                    for i, r in enumerate(results)
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Top pick highlight
            best = results[0]
            st.success(
                f"### 🥇 המלצה מובילה: **{best['ticker']}**\n\n"
                f"ציון: `{best['investment_score']:.5f}` | "
                f"תשואה צפויה: `{best['expected_return'] * 100:.1f}%` | "
                f"מגמה: {trend_emoji.get(best['trend'], best['trend'])}"
            )

            # Formula explainer
            with st.expander("📐 הסבר הנוסחה המתמטית"):
                st.latex(
                    r"Score = \frac{Probability_{Profit} \cdot Expected_{Return}}{Risk_{Factor}}"
                )
                st.markdown(f"""
**פירוט עבור {best['ticker']}:**

| מרכיב | ערך | פירוש |
|---|---|---|
| הסתברות רווח | `{best['probability_profit']}` | RSI + MACD + מגמה |
| תשואה צפויה | `{best['expected_return']:.4f}` ({best['expected_return']*100:.1f}%) | מחיר יעד מול מחיר נוכחי |
| מקדם סיכון | `{best['risk_factor']}` | תנודתיות × רמת סיכון משתמש |
| **ציון סופי** | **`{best['investment_score']:.5f}`** | ↑ גבוה יותר = טוב יותר |
                """)

# ── Tab 3: Market Chart ────────────────────────────────────────────────────────
with tab_chart:
    st.subheader("📈 גרף נרות יפני")

    col_t, col_p = st.columns([2, 1])
    with col_t:
        chart_ticker = st.text_input("טיקר:", value="SPY", key="chart_ticker_input")
    with col_p:
        period_options = {
            "שבוע": "5d",
            "חודש": "1mo",
            "3 חודשים": "3mo",
            "6 חודשים": "6mo",
            "שנה": "1y",
        }
        period_label = st.selectbox("תקופה:", list(period_options.keys()), index=2)

    if st.button("📊 הצג גרף", type="primary"):
        with st.spinner("טוען נתוני שוק..."):
            try:
                data = get_historical_data(chart_ticker.upper(), period_options[period_label])

                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=data.index,
                            open=data["Open"],
                            high=data["High"],
                            low=data["Low"],
                            close=data["Close"],
                            increasing_line_color="#26a69a",
                            decreasing_line_color="#ef5350",
                            name=chart_ticker.upper(),
                        )
                    ]
                )

                # 20-day MA overlay
                ma20 = data["Close"].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma20,
                        name="ממוצע נע 20",
                        line=dict(color="orange", width=1.5, dash="dot"),
                    )
                )

                fig.update_layout(
                    title=f"{chart_ticker.upper()} — {period_label}",
                    yaxis_title="מחיר ($)",
                    xaxis_title="תאריך",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                current_p = float(data["Close"].iloc[-1])
                start_p = float(data["Close"].iloc[0])
                change_pct = (current_p - start_p) / start_p * 100
                high_p = float(data["High"].max())
                low_p = float(data["Low"].min())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("מחיר נוכחי", f"${current_p:.2f}")
                m2.metric("שינוי בתקופה", f"{change_pct:+.2f}%", delta=f"{change_pct:+.2f}%")
                m3.metric("שיא התקופה", f"${high_p:.2f}")
                m4.metric("שפל התקופה", f"${low_p:.2f}")

            except Exception as exc:
                st.error(f"❌ שגיאה בטעינת {chart_ticker}: {exc}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **אזהרה חשובה:** המידע המוצג הוא לצורכי מידע בלבד ואינו מהווה ייעוץ פיננסי מוסמך. "
    "השקעות בשוק ההון כרוכות בסיכון אובדן הון. | Pro-Investor AI © 2025"
)
