"""
Pro-Investor — Streamlit UI
Spotify/Netflix dark aesthetic · zero-scroll layout · intraday charts.
Code/comments in English; all UI text and AI responses in Hebrew.

Execution order (critical for login gate):
  1. Page config
  2. Session-state defaults          ← must come before ANY rendering
  3. GROQ key + heavy imports
  4. LOGIN GATEKEEPER                ← injects login CSS, renders form, st.stop()
  5. Dashboard CSS injection         ← only reached when logged_in is True
  6. Sidebar → Tabs → Chat / Analyze / Chart / History
"""

import json
import os
import socket
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))


# ── BackendClient ──────────────────────────────────────────────────────────────
class BackendClient:
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

    def ping(self) -> bool:
        return self._send({"action": "ping"}) is not None

    def score(self, ticker: str, risk: str, months: int) -> dict | None:
        return self._send({"action": "score", "ticker": ticker, "risk": risk, "months": months})


# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Pro-Investor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults (BEFORE any rendering or CSS) ──────────────────────
for _k, _default in {
    "logged_in":    False,
    "user_id":      None,
    "user_name":    "",
    "user_profile": {},
    "agent":        None,
    "chat_history": [],
    "backend":      BackendClient(),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _default

# ── GROQ key ───────────────────────────────────────────────────────────────────
_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if _key:
    os.environ["GROQ_API_KEY"] = _key
else:
    st.error("### ⚠️ מפתח Groq API חסר — הוסף ל-.streamlit/secrets.toml")
    st.stop()

# ── Heavy imports (after GROQ key check so failures are explicit) ──────────────
from agent import InvestmentAgent
from data_engine import get_historical_data, get_multiple_prices
from scoring_engine import score_ticker
from database import (
    init_db, authenticate_user, create_user, update_user_profile,
    save_recommendation, get_recent_recommendations, get_stats,
)

try:
    init_db()
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN GATEKEEPER
# Injected BEFORE the dashboard CSS so the flex-chain never clips the form.
# ══════════════════════════════════════════════════════════════════════════════
_LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {
    font-family: 'Heebo', 'Arial', sans-serif !important;
    box-sizing: border-box;
}

/* Login page: dark bg, natural scroll, no flex constraints */
html, body, .stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
    direction: rtl !important;
    min-height: 100vh !important;
}

/* Radial green glow at top */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse 80% 38% at 50% 0%,
        rgba(29,185,84,0.10) 0%, transparent 62%);
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"], [data-testid="stDecoration"],
.stDeployButton, #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="stSidebarResizeHandle"] {
    display: none !important;
}

/* Block-container: centered, max-width 520px */
.block-container {
    max-width: 100% !important;
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    overflow: visible !important;
}
section.main, [data-testid="stMain"] {
    padding-top: 0 !important;
    overflow: visible !important;
}

/* RTL text */
body, p, span, div, li, h1, h2, h3, h4,
label, button, input, textarea, select,
[data-testid="stWidgetLabel"] {
    direction: rtl !important;
    text-align: right !important;
}

/* Login hero */
.login-hero {
    text-align: center !important;
    padding: 44px 0 32px !important;
    direction: rtl !important;
    position: relative; z-index: 1;
}
.login-hero h1 {
    color: #1DB954 !important;
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    letter-spacing: -1.2px !important;
    margin: 0 !important;
}
.login-hero p {
    color: rgba(255,255,255,0.32) !important;
    font-size: 0.92rem !important;
    margin-top: 8px !important;
}
.login-foot {
    text-align: center !important;
    color: rgba(255,255,255,0.18) !important;
    font-size: 0.68rem !important;
    margin-top: 30px !important;
    direction: rtl !important;
}

/* Glass form card */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    padding: 22px 18px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    position: relative; z-index: 1;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stPasswordInput input {
    background: #141414 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    direction: rtl !important;
    font-size: 0.92rem !important;
    padding: 9px 12px !important;
    transition: border-color 0.15s !important;
}
.stTextInput input:focus, .stPasswordInput input:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.14) !important;
    background: #1a1a1a !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #141414 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.9rem !important;
}

/* Form submit + regular buttons */
[data-testid="stFormSubmitButton"] > button,
.stButton > button {
    background: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
.stButton > button:hover {
    background: #1ed760 !important;
    box-shadow: 0 0 18px rgba(29,185,84,0.38) !important;
}

/* Alerts RTL */
.stAlert, [data-testid="stAlert"] {
    direction: rtl !important;
    text-align: right !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
}
.stAlert > div, .stAlert p, .stAlert span { direction: rtl !important; }

/* Tabs (login / signup tabs) */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 0.6rem !important;
}
.stTabs [data-baseweb="tab"] {
    color: #727272 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 20px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] { direction: ltr !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #1DB954 !important; }

/* Progress */
.stProgress > div > div > div { background: #1DB954 !important; }
</style>
"""


def _render_login_screen() -> None:
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="login-hero">
        <div style="font-size:2.8rem; margin-bottom:10px;">📈</div>
        <h1>Pro-Investor</h1>
        <p>יועץ ההשקעות החכם שלך — מחפש, מנתח ומדרג בזמן אמת</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        tab_login, tab_signup = st.tabs(["🔐 התחברות", "📝 הרשמה"])

        # ── Login ──────────────────────────────────────────────────────────────
        with tab_login:
            with st.form("login_form"):
                name      = st.text_input("👤 שם משתמש")
                password  = st.text_input("🔑 סיסמה", type="password")
                submitted = st.form_submit_button("התחבר →")
            if submitted:
                if not name or not password:
                    st.error("יש למלא שם משתמש וסיסמה")
                else:
                    profile = authenticate_user(name.strip(), password)
                    if profile:
                        st.session_state.logged_in    = True
                        st.session_state.user_id      = profile["id"]
                        st.session_state.user_name    = profile["name"]
                        st.session_state.user_profile = profile
                        st.session_state.agent        = InvestmentAgent()
                        st.rerun()
                    else:
                        st.error("שם משתמש או סיסמה שגויים")

        # ── Signup ─────────────────────────────────────────────────────────────
        with tab_signup:
            with st.form("signup_form"):
                nn  = st.text_input("👤 שם משתמש חדש")
                np  = st.text_input("🔑 סיסמה", type="password")
                np2 = st.text_input("🔑 אימות סיסמה", type="password")
                nb  = st.number_input("💰 תקציב ($)", min_value=100, max_value=10_000_000, value=10_000, step=500)
                ro  = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
                rl  = st.selectbox("⚖️ רמת סיכון", list(ro.keys()), index=1)
                nd  = st.slider("📅 אופק השקעה (חודשים)", 1, 60, 12)
                rs  = st.form_submit_button("הרשמה →")
            if rs:
                if not nn or not np:
                    st.error("שם משתמש וסיסמה הם שדות חובה")
                elif np != np2:
                    st.error("הסיסמאות אינן תואמות")
                else:
                    try:
                        create_user(nn.strip(), np, float(nb), ro[rl], int(nd))
                        st.success(f"✅ נרשמת בהצלחה, {nn.strip()}! עבור ללשונית התחברות")
                    except ValueError as exc:
                        st.error(str(exc))

    st.markdown(
        '<div class="login-foot">⚠️ המידע לצורכי מידע בלבד ואינו ייעוץ פיננסי מוסמך</div>',
        unsafe_allow_html=True,
    )


# ── GATEKEEPER: if not authenticated, show login and halt execution ─────────
if not st.session_state.logged_in:
    _render_login_screen()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CSS — only injected for authenticated users
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Heebo Hebrew font ── */
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800;900&display=swap');

/* ── HARD RESET ── */
*, *::before, *::after {
    font-family: 'Heebo', 'Arial', sans-serif !important;
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

/* ── Base colors + ABSOLUTE NO-SCROLL root ── */
html, body {
    background-color: #000000 !important;
    color: #ffffff !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}
.stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: row !important;
}

/* ── Hide Streamlit chrome (header, footer, deploy button) ── */
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton              { display: none !important; }
#MainMenu                    { display: none !important; }
footer                       { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

/* ── Main content: flex-column fills remaining viewport height ── */
section.main, [data-testid="stMain"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    padding-top: 0 !important;
}
.block-container {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
    max-width: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

/* ── RTL — every text node ── */
.stApp { direction: rtl !important; }
body, p, span, div, li, h1, h2, h3, h4, h5, h6,
label, button, input, textarea, select,
.stMarkdown, .stMarkdown *,
[data-testid="stText"],
[data-testid="stWidgetLabel"] {
    direction: rtl !important;
    text-align: right !important;
}

/* ── Sidebar: absolute overflow lockdown + fixed width ── */
section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-left: 1px solid rgba(255,255,255,0.06) !important;
    border-right: none !important;
    width: 230px !important;
    min-width: 230px !important;
    max-width: 230px !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    flex-shrink: 0 !important;
}
/* All descendant containers: width 100% + box-sizing, no horizontal push */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    min-width: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] * {
    direction: rtl !important;
    text-align: right !important;
    box-sizing: border-box !important;
    max-width: 100% !important;
}
/* Collapse sidebar widget margins */
section[data-testid="stSidebar"] .element-container {
    margin-bottom: 0.12rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.22rem 0 !important;
}
/* Compact labels */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.68rem !important;
    color: #727272 !important;
    margin-bottom: 0 !important;
    line-height: 1.2 !important;
}
/* Compact inputs + selects + slider */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] {
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    font-size: 0.76rem !important;
    padding: 3px 8px !important;
    height: 28px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] {
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    font-size: 0.76rem !important;
    padding: 2px 6px !important;
    min-height: 28px !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-top: 0.1rem !important;
}
/* Kill the drag-resize handle — sidebar width is fixed */
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* ── Tabs (underline, Spotify-style) ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    color: #727272 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }

/* ── Reduce element-container gaps ── */
.element-container { margin-bottom: 0.35rem !important; }
.stMarkdown p      { margin-bottom: 0.25rem !important; }
hr                 { margin: 0.4rem 0 !important; border-color: rgba(255,255,255,0.06) !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.15s ease !important;
    width: auto !important;
}
.stButton > button:hover {
    background: #1ed760 !important;
    transform: scale(1.03) !important;
    box-shadow: 0 0 20px rgba(29,185,84,0.4) !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input,
.stChatInput textarea, .stPasswordInput input {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    direction: rtl !important;
    font-size: 0.9rem !important;
    padding: 8px 12px !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stPasswordInput input:focus, .stChatInput textarea:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.15) !important;
    background: #1f1f1f !important;
}
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.88rem !important;
}

/* ── Alerts — forced RTL ── */
.stAlert, [data-testid="stAlert"],
[data-baseweb="notification"],
div[class*="alert"], div[class*="Alert"] {
    direction: rtl !important;
    text-align: right !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    padding: 8px 12px !important;
}
.stAlert > div, .stAlert p, .stAlert span { direction: rtl !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    margin-bottom: 8px !important;
    direction: rtl !important;
    padding: 10px 16px !important;
    border: 1px solid transparent !important;
    isolation: isolate !important;
    position: relative !important;
    z-index: 1 !important;
    overflow: hidden !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.07) !important;
    margin-left: 8% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(29,185,84,0.06) !important;
    border-color: rgba(29,185,84,0.15) !important;
    margin-right: 4% !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    direction: rtl !important;
    text-align: right !important;
}

/* ── Metrics — compact ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="metric-container"]:hover { border-color: rgba(29,185,84,0.25) !important; }
[data-testid="stMetricValue"] {
    color: #1DB954 !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] { color: #727272 !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"]  { font-size: 0.72rem !important; }

/* ── Forms ── */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}

/* ── Expander — compact ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: #727272 !important;
    direction: rtl !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    font-size: 0.8rem !important;
    padding: 8px 12px !important;
    min-height: unset !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 0 0 8px 8px !important;
    padding: 10px 12px !important;
}
.streamlit-expanderHeader *, .streamlit-expanderContent * { direction: rtl !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"]  { direction: ltr !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #1DB954 !important; }

/* ── DataTable ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Progress / Spinner ── */
.stProgress > div > div > div { background: #1DB954 !important; }
.stSpinner > div { border-top-color: #1DB954 !important; }

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    direction: rtl !important;
    background: rgba(29,185,84,0.05) !important;
    border: 1px solid rgba(29,185,84,0.15) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
}

/* ── Command-center header ── */
.cmd-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    direction: rtl;
    padding: 0 0 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 6px;
    flex-shrink: 0;
}
.cmd-title { font-size: 1.35rem; font-weight: 900; color: #ffffff; letter-spacing: -0.3px; }
.cmd-sub   { font-size: 0.7rem; color: #3d3d3d; font-weight: 500; }

/* ── Compact live ticker grid (sidebar) ── */
.ticker-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    direction: rtl;
    margin-top: 4px;
}
.ticker-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 5px 9px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.t-sym   { font-size: 0.62rem; font-weight: 700; color: #535353; letter-spacing: 0.3px; }
.t-price { font-size: 0.76rem; font-weight: 700; color: #1DB954; }

/* ── Sidebar brand + user block ── */
.sidebar-brand {
    font-size: 1.1rem;
    font-weight: 900;
    color: #1DB954;
    letter-spacing: -0.3px;
    direction: rtl;
    padding: 4px 0 2px;
}
.sidebar-user {
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff;
    direction: rtl;
}
.sidebar-section {
    font-size: 0.6rem;
    font-weight: 700;
    color: #3d3d3d;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    direction: rtl;
    padding: 4px 0 3px;
}
.server-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 500px;
    font-size: 0.65rem;
    font-weight: 600;
    direction: rtl;
}
.server-pill.on  { background: rgba(29,185,84,0.1); color: #1DB954; border: 1px solid rgba(29,185,84,0.2); }
.server-pill.off { background: rgba(255,255,255,0.04); color: #535353; border: 1px solid rgba(255,255,255,0.08); }

/* ── Verified advisor badge ── */
.verified-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.28);
    border-radius: 500px;
    padding: 2px 10px;
    font-size: 0.67rem;
    font-weight: 600;
    color: #1DB954;
    margin-bottom: 8px;
    letter-spacing: 0.2px;
}

/* ── Stock media cards ── */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 12px;
    direction: rtl;
    margin-top: 14px;
}
.stock-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    direction: rtl;
    text-align: right;
    position: relative;
    overflow: hidden;
}
.stock-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1DB954, transparent);
    opacity: 0;
    transition: opacity 0.22s;
}
.stock-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(29,185,84,0.32);
    box-shadow: 0 12px 40px rgba(29,185,84,0.1), 0 4px 12px rgba(0,0,0,0.6);
    background: rgba(29,185,84,0.05);
}
.stock-card:hover::before { opacity: 1; }
.stock-card.card-avoid:hover {
    border-color: rgba(229,9,20,0.3);
    box-shadow: 0 12px 40px rgba(229,9,20,0.08), 0 4px 12px rgba(0,0,0,0.6);
    background: rgba(229,9,20,0.04);
}
.stock-card.card-avoid::before { background: linear-gradient(90deg, #E50914, transparent); }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.card-ticker { font-size: 1.35rem; font-weight: 900; color: #fff; letter-spacing: -0.3px; line-height: 1; }
.card-rank   { font-size: 0.62rem; color: #3d3d3d; margin-top: 2px; }
.card-badge  { font-size: 0.67rem; font-weight: 700; padding: 4px 10px; border-radius: 500px; white-space: nowrap; }
.badge-buy   { background: rgba(29,185,84,0.15);  color: #1DB954; border: 1px solid rgba(29,185,84,0.35); }
.badge-wait  { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); }
.badge-avoid { background: rgba(229,9,20,0.15);   color: #E50914; border: 1px solid rgba(229,9,20,0.35); }
.card-price  { font-size: 1.7rem; font-weight: 800; color: #fff; letter-spacing: -0.8px; line-height: 1; margin-bottom: 2px; }
.card-target { font-size: 0.72rem; color: #3d3d3d; margin-bottom: 12px; }
.card-divider{ height: 1px; background: rgba(255,255,255,0.06); margin-bottom: 10px; }
.card-metrics{ margin-bottom: 10px; }
.metric-row  { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; }
.metric-label{ font-size: 0.72rem; color: #535353; }
.metric-value{ font-size: 0.8rem; font-weight: 600; color: #fff; }
.val-green   { color: #1DB954 !important; }
.val-red     { color: #E50914 !important; }
.val-amber   { color: #fbbf24 !important; }
.card-footer { font-size: 0.72rem; color: #3d3d3d; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); }
.champion-card {
    background: linear-gradient(135deg, rgba(29,185,84,0.1), rgba(29,185,84,0.04));
    border: 1px solid rgba(29,185,84,0.35);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    direction: rtl;
    text-align: right;
}
.champion-label  { font-size: 0.65rem; font-weight: 700; color: #1DB954; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
.champion-name   { font-size: 2rem; font-weight: 900; color: #fff; letter-spacing: -0.8px; margin-bottom: 4px; }
.champion-detail { font-size: 0.85rem; color: #b3b3b3; }

/* ── Loading animation ── */
@keyframes scan-bar { 0% { left: -60%; } 100% { left: 110%; } }
@keyframes pulse-txt { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.loader-wrap { padding: 12px 0; direction: rtl; }
.loader-bar  { height: 2px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; margin: 10px 0; position: relative; }
.loader-bar::after {
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, #1DB954, transparent);
    animation: scan-bar 1.2s linear infinite;
}
.loader-txt { color: #1DB954; font-size: 0.82rem; font-weight: 600; animation: pulse-txt 1.3s ease-in-out infinite; direction: rtl; text-align: right; }

/* ── Chart interval badge ── */
.interval-badge {
    display: inline-block;
    background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #1DB954;
    margin-right: 6px;
}

/* ── ANTI-OVERLAP: Stacking context isolation ── */
[data-testid="stMarkdownContainer"] {
    position: relative !important;
    z-index: 1 !important;
    isolation: isolate !important;
}
[data-testid="stStatus"], .stStatus,
[data-testid="stStatusWidget"] {
    isolation: isolate !important;
    position: relative !important;
    z-index: 0 !important;
    overflow: hidden !important;
}

/* ── Sidebar final lock: stSidebarUserContent ── */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    overflow: hidden !important;
    height: auto !important;
    padding-bottom: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   FLEX CHAIN: tabs expand to fill remaining height
   Tab panels get their OWN internal scroll (no page scroll at all)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTabs {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
.stTabs [data-baseweb="tab-list"] {
    flex-shrink: 0 !important;
}
/* The ONLY scroll surface: each tab panel */
.stTabs [data-baseweb="tab-panel"],
[data-baseweb="tab-panel"],
[data-testid="stTabContent"] {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(255,255,255,0.08) transparent !important;
    padding-bottom: 4px !important;
}
[data-baseweb="tab-panel"]::-webkit-scrollbar { width: 4px !important; }
[data-baseweb="tab-panel"]::-webkit-scrollbar-track { background: transparent !important; }
[data-baseweb="tab-panel"]::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 2px !important;
}

/* ── Chat input: sticky at bottom of the scrollable tab panel ── */
[data-testid="stBottom"] {
    position: sticky !important;
    bottom: 0 !important;
    z-index: 200 !important;
    background: #000000 !important;
    padding: 3px 0 !important;
    flex-shrink: 0 !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Plotly chart: never expand beyond its container ── */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

/* ── RTL: exhaustive [data-testid="stMarkdownContainer"] targeting ── */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] li,  [data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol,  [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th,  [data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,  [data-testid="stMarkdownContainer"] a {
    direction: rtl !important;
    text-align: right !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    direction: rtl !important;
    text-align: right !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
    direction: rtl !important;
    text-align: right !important;
}
[data-testid="stAlert"] p, [data-testid="stAlert"] span,
[data-testid="stAlert"] div, [data-testid="stAlert"] * {
    direction: rtl !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

# ── Agent init (only for authenticated users) ──────────────────────────────────
if st.session_state.agent is None:
    st.session_state.agent = InvestmentAgent()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — compact command-center style
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📈 Pro-Investor</div>', unsafe_allow_html=True)

    _ok = st.session_state.backend.ping()
    st.markdown(
        f'<div class="sidebar-user">{st.session_state.user_name}'
        f'&nbsp;<span class="server-pill {"on" if _ok else "off"}">{"🟢 Online" if _ok else "⚪ Local"}</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("🚪 יציאה", key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()

    st.markdown('<div class="sidebar-section">💼 פרופיל</div>', unsafe_allow_html=True)
    _prof   = st.session_state.user_profile
    budget  = st.number_input("תקציב ($)", min_value=100, max_value=10_000_000,
                               value=int(_prof.get("budget", 10_000)), step=500, format="%d")
    risk_map   = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
    _ri        = {"low": 0, "medium": 1, "high": 2}.get(_prof.get("risk_tolerance", "medium"), 1)
    risk_label = st.selectbox("סיכון", list(risk_map.keys()), index=_ri)
    risk       = risk_map[risk_label]
    duration   = st.slider("אופק (חודשים)", 1, 60, int(_prof.get("duration_months", 12)))

    if st.button("💾 שמור", key="save_profile"):
        try:
            update_user_profile(st.session_state.user_id, float(budget), risk, int(duration))
            st.session_state.user_profile.update({
                "budget": float(budget), "risk_tolerance": risk, "duration_months": int(duration)
            })
            st.success("✅ נשמר")
        except Exception as exc:
            st.error(f"שגיאה: {exc}")

    st.divider()

    st.markdown('<div class="sidebar-section">📡 שוק · עכשיו</div>', unsafe_allow_html=True)
    try:
        prices = get_multiple_prices(["SPY", "AAPL", "NVDA", "BTC-USD"])
        html = '<div class="ticker-grid">'
        for sym, price in prices.items():
            pstr = f"${price:,.2f}" if price else "—"
            html += f'<div class="ticker-item"><span class="t-sym">{sym}</span><span class="t-price">{pstr}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.caption("נתונים זמינים בצ'אט")

    st.divider()
    st.markdown(
        "<div style='font-size:0.62rem; color:#2a2a2a; line-height:1.9; direction:rtl;'>"
        "🤖 LLaMA-3.3-70b · Groq<br>🗄️ ChromaDB · Sentence-T<br>🔍 DuckDuckGo</div>",
        unsafe_allow_html=True,
    )

# ── Compact command-center header ──────────────────────────────────────────────
st.markdown(
    '<div class="cmd-header">'
    '<span class="cmd-title">📈 Pro-Investor</span>'
    '<span class="cmd-sub">Command Center &nbsp;·&nbsp; סוכן AI &nbsp;·&nbsp; זמן אמת</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_analyze, tab_chart, tab_history = st.tabs([
    "💬 ייעוץ AI", "📊 ניתוח שוק", "📈 גרפים", "📋 היסטוריה"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="📈"):
                st.markdown('<div class="verified-badge">📈 Pro-Investor AI · יועץ מאומת</div>',
                            unsafe_allow_html=True)
                st.markdown(msg["content"])
                log = msg.get("tool_log", [])
                if log:
                    with st.expander(f"🔍 מחקר הסוכן ({len(log)} צעדים)", expanded=False):
                        for step in log:
                            st.markdown(f"{step['icon']} **{step['label']}** — {step['summary']}")

    user_input = st.chat_input("שאל את הסוכן... (לדוגמה: 'נתח NVDA' או 'מה מצב BTC?')")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        profile = {
            "budget": budget, "risk_tolerance": risk,
            "duration_months": duration, "user_name": st.session_state.user_name,
        }

        with st.chat_message("assistant", avatar="📈"):
            loader_ph = st.empty()
            loader_ph.markdown("""
            <div class="loader-wrap">
                <div class="loader-bar"></div>
                <div class="loader-txt">מחפש בשווקים עבורך...</div>
            </div>""", unsafe_allow_html=True)

            try:
                answer, tool_log = st.session_state.agent.run(user_input, profile)
                loader_ph.empty()

                st.markdown('<div class="verified-badge">📈 Pro-Investor AI · יועץ מאומת</div>',
                            unsafe_allow_html=True)
                st.markdown(answer)

                if tool_log:
                    with st.expander(f"🔍 מחקר הסוכן ({len(tool_log)} צעדים)", expanded=False):
                        for step in tool_log:
                            st.markdown(f"{step['icon']} **{step['label']}** — {step['summary']}")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "tool_log": tool_log}
                )

                try:
                    for step in tool_log:
                        if step.get("tool") == "get_investment_score" and step.get("data"):
                            d = step["data"]
                            save_recommendation(
                                ticker=d.get("ticker", ""),
                                verdict=d.get("human", {}).get("verdict_raw", "WAIT"),
                                price=float(d.get("current_price", 0)),
                                score_data=d, ai_response=answer,
                                user_id=st.session_state.user_id,
                            )
                            break
                except Exception:
                    pass

            except Exception as exc:
                loader_ph.empty()
                _e = str(exc).lower()
                if "429" in _e or "rate_limit" in _e or "rate limit" in _e or "too many" in _e:
                    err = "מצטער, הגעתי למכסת הבקשות לדקה. אנא נסה שוב בעוד מספר שניות. ⏱️"
                    st.warning(err)
                else:
                    err = f"אני מצטער, נתקלתי בשגיאה טכנית: {exc}"
                    st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err, "tool_log": []})

    if st.session_state.chat_history:
        if st.button("🗑️ נקה שיחה"):
            st.session_state.chat_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Analysis (Media Cards)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        tickers_raw = st.text_input("טיקרים (פסיק):", value="AAPL, MSFT, NVDA, SPY, BTC-USD",
                                    label_visibility="collapsed")
    with col_btn:
        run_analysis = st.button("🔍 נתח", key="analyze_btn")

    if run_analysis and tickers_raw:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        results, bar = [], st.progress(0, text="מנתח...")

        for i, sym in enumerate(tickers):
            try:
                results.append(score_ticker(sym, risk, duration))
            except Exception as e:
                st.warning(f"⚠️ {sym}: {e}")
            bar.progress((i + 1) / len(tickers), text=f"מחשב ציון ל-{sym}...")

        bar.empty()

        if results:
            results.sort(key=lambda x: x["investment_score"], reverse=True)

            TREND_MAP = {"bullish": "📈 עולה", "bearish": "📉 יורד", "neutral": "➡️ ניטרלי"}
            def verdict(s):
                return ("buy","✅ קנייה") if s >= 0.28 else ("avoid","❌ הימנע") if s < 0.05 else ("wait","⚠️ המתן")

            best = results[0]
            bvc, bvt = verdict(best["investment_score"])
            st.markdown(f"""
            <div class="champion-card">
                <div class="champion-label">🥇 המלצה מובילה</div>
                <div class="champion-name">{best['ticker']}</div>
                <div class="champion-detail">
                    <span class="card-badge badge-{bvc}">{bvt}</span>&nbsp;
                    ${best['current_price']:.2f} → <span style="color:#1DB954">${best['target_price']:.2f}</span>
                    &nbsp;·&nbsp; {TREND_MAP.get(best['trend'], best['trend'])}
                </div>
            </div>""", unsafe_allow_html=True)

            grid = '<div class="cards-grid">'
            for rank, r in enumerate(results):
                ret    = r["expected_return"] * 100
                vc, vt = verdict(r["investment_score"])
                rc     = "val-green" if ret > 0 else "val-red"
                rsic   = "val-green" if r["rsi"] < 45 else ("val-red" if r["rsi"] > 65 else "")
                avoid  = "card-avoid" if vc == "avoid" else ""
                grid  += f"""
                <div class="stock-card {avoid}">
                    <div class="card-header">
                        <div><div class="card-ticker">{r['ticker']}</div><div class="card-rank">#{rank+1}</div></div>
                        <span class="card-badge badge-{vc}">{vt}</span>
                    </div>
                    <div class="card-price">${r['current_price']:.2f}</div>
                    <div class="card-target">יעד: ${r['target_price']:.2f}</div>
                    <div class="card-divider"></div>
                    <div class="card-metrics">
                        <div class="metric-row"><span class="metric-label">תשואה</span><span class="metric-value {rc}">{ret:+.1f}%</span></div>
                        <div class="metric-row"><span class="metric-label">P(רווח)</span><span class="metric-value">{r['probability_profit']*100:.0f}%</span></div>
                        <div class="metric-row"><span class="metric-label">RSI</span><span class="metric-value {rsic}">{r['rsi']:.1f}</span></div>
                        <div class="metric-row"><span class="metric-label">תנודתיות</span><span class="metric-value">{r['volatility']*100:.1f}%</span></div>
                    </div>
                    <div class="card-footer">{TREND_MAP.get(r['trend'], r['trend'])}</div>
                </div>"""
            grid += '</div>'
            st.markdown(grid, unsafe_allow_html=True)

            with st.expander("📐 נוסחת הציון"):
                st.latex(r"Score = \frac{P_{Profit} \cdot Expected_{Return}}{Risk_{Factor}}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — High-Resolution Intraday Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    CHART_CONFIGS = {
        "יום  · 5 דק'":   ("1d",  "5m",  "5 דקות"),
        "שבוע · 15 דק'":  ("5d",  "15m", "15 דקות"),
        "חודש · שעה":     ("1mo", "1h",  "שעה"),
        "3 חודשים · יום": ("3mo", "1d",  "יום"),
        "6 חודשים · יום": ("6mo", "1d",  "יום"),
        "שנה · יום":      ("1y",  "1d",  "יום"),
    }

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        chart_sym  = st.text_input("טיקר:", value="SPY", key="chart_sym", label_visibility="collapsed")
    with c2:
        period_lbl = st.selectbox("תקופה:", list(CHART_CONFIGS.keys()), index=0, label_visibility="collapsed")
    with c3:
        show_chart = st.button("📊 הצג", key="chart_btn")

    if show_chart:
        period, interval, interval_label = CHART_CONFIGS[period_lbl]
        with st.spinner("טוען נתונים..."):
            try:
                data = get_historical_data(chart_sym.upper(), period=period, interval=interval)

                intraday = interval in ("1m", "2m", "5m", "15m", "30m", "1h")
                x_fmt    = "%H:%M" if intraday else "%d/%m"

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data["Open"], high=data["High"],
                    low=data["Low"],   close=data["Close"],
                    increasing=dict(line=dict(color="#1DB954", width=1), fillcolor="#1DB954"),
                    decreasing=dict(line=dict(color="#E50914", width=1), fillcolor="#E50914"),
                    name=chart_sym.upper(),
                    hovertext=data.index.strftime("%H:%M %d/%m" if intraday else "%d/%m/%y"),
                ))

                ma = data["Close"].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=ma,
                    name="ממוצע נע 20",
                    line=dict(color="rgba(251,191,36,0.8)", width=1.2, dash="dot"),
                    hovertemplate="%{y:$.2f}<extra>MA20</extra>",
                ))

                fig.add_trace(go.Bar(
                    x=data.index, y=data["Volume"],
                    name="נפח",
                    marker_color=[
                        "#1DB954" if c >= o else "#E50914"
                        for c, o in zip(data["Close"], data["Open"])
                    ],
                    opacity=0.25,
                    yaxis="y2",
                    showlegend=False,
                    hovertemplate="%{y:,.0f}<extra>נפח</extra>",
                ))

                cur = float(data["Close"].iloc[-1])
                beg = float(data["Close"].iloc[0])
                pct = (cur - beg) / beg * 100
                hi  = float(data["High"].max())
                lo  = float(data["Low"].min())

                fig.update_layout(
                    title=dict(
                        text=(
                            f"<b>{chart_sym.upper()}</b>"
                            f"  <span style='font-size:13px; color:{'#1DB954' if pct>=0 else '#E50914'}'>{pct:+.2f}%</span>"
                            f"  <span style='font-size:11px; color:#535353'>· {interval_label} / נר</span>"
                        ),
                        x=0, font=dict(size=16, color="#ffffff"),
                    ),
                    xaxis=dict(
                        tickformat=x_fmt,
                        gridcolor="rgba(255,255,255,0.04)",
                        linecolor="rgba(255,255,255,0.06)",
                        rangeslider=dict(visible=False),
                        showgrid=True,
                    ),
                    yaxis=dict(
                        title="מחיר ($)",
                        gridcolor="rgba(255,255,255,0.04)",
                        linecolor="rgba(255,255,255,0.06)",
                        side="right",
                        tickformat="$.2f",
                        domain=[0.2, 1.0],
                    ),
                    yaxis2=dict(
                        domain=[0.0, 0.17],
                        showticklabels=False,
                        gridcolor="rgba(255,255,255,0.02)",
                    ),
                    legend=dict(
                        orientation="h", y=1.02, x=1, xanchor="right",
                        font=dict(size=11, color="#727272"),
                        bgcolor="transparent",
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#000000",
                    plot_bgcolor="#070707",
                    font=dict(family="Heebo", color="#b3b3b3", size=11),
                    height=430,
                    margin=dict(l=10, r=10, t=44, b=10),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#1a1a1a",
                        bordercolor="rgba(255,255,255,0.12)",
                        font=dict(family="Heebo", size=11, color="#ffffff"),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("מחיר נוכחי",  f"${cur:,.2f}")
                m2.metric("שינוי תקופה", f"{pct:+.2f}%", delta=f"{pct:+.2f}%")
                m3.metric("שיא תקופה",   f"${hi:,.2f}")
                m4.metric("שפל תקופה",   f"${lo:,.2f}")

                st.markdown(
                    f'<span class="interval-badge">נר: {interval_label}</span>'
                    f'<span style="font-size:0.7rem; color:#3d3d3d;">'
                    f'{len(data):,} נרות · {period_lbl}</span>',
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.error(f"❌ שגיאה: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Recommendation History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    if st.button("🔄 רענן", key="refresh_history"):
        st.rerun()

    try:
        stats = get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("סה\"כ", stats["total"])
        c2.metric("✅ קנייה", stats["buys"])
        c3.metric("⚠️ המתן", stats["waits"])
        c4.metric("❌ הימנע", stats["avoids"])
        st.divider()

        recs = get_recent_recommendations(50)
        if recs:
            VERDICT_HE = {"BUY": "✅ קנייה", "WAIT": "⚠️ המתן", "AVOID": "❌ הימנע"}
            st.dataframe(
                pd.DataFrame([{
                    "תאריך":  r["date"],
                    "טיקר":   r["ticker"],
                    "פסיקה":  VERDICT_HE.get(r["verdict"], r["verdict"]),
                    "מחיר":   f"${r['price']:.2f}" if r["price"] else "—",
                    "ציון": (
                        "גבוה ✅" if (r["score"] or 0) >= 0.28
                        else "בינוני 🟡" if (r["score"] or 0) >= 0.10
                        else "נמוך 🔴"
                    ) if r["score"] is not None else "—",
                    "תשואה":  f"{r['return']*100:.1f}%" if r["return"] else "—",
                } for r in recs]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("📭 אין המלצות עדיין — שאל את הסוכן על מניה")
    except Exception as exc:
        st.error(f"❌ שגיאת DB: {exc}")

# ── Footer (minimal) ───────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#1a1a1a; font-size:0.65rem; direction:rtl; margin-top:4px;'>"
    "Pro-Investor AI © 2026 · מידע לצורכי מידע בלבד</p>",
    unsafe_allow_html=True,
)
