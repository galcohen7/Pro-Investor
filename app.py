"""
Pro-Investor — Streamlit UI
Spotify/Netflix premium dark aesthetic.
Pure UI overhaul — all backend logic (agent, sockets, DB) unchanged.
Code/comments in English; all UI text and AI responses in Hebrew.
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


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pro-Investor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GROQ key ───────────────────────────────────────────────────────────────────
_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if _key:
    os.environ["GROQ_API_KEY"] = _key
else:
    st.error("### ⚠️ מפתח Groq API חסר — הוסף ל-.streamlit/secrets.toml")
    st.stop()

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
# PREMIUM DARK CSS — Spotify/Netflix aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts: Heebo (modern Hebrew) ── */
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {
    font-family: 'Heebo', 'Arial', sans-serif !important;
    box-sizing: border-box;
}
html, body, .stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Strict RTL ── */
.stApp { direction: rtl !important; }
body, p, span, div, li, h1, h2, h3, h4, h5, h6,
label, button, input, textarea, select,
.stMarkdown, .stMarkdown *,
[data-testid="stText"],
[data-testid="stWidgetLabel"] {
    direction: rtl !important;
    text-align: right !important;
}

/* ── Sidebar: Spotify #121212 ── */
section[data-testid="stSidebar"] {
    background: #121212 !important;
    border-left: 1px solid rgba(255,255,255,0.07) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    direction: rtl !important;
    text-align: right !important;
}

/* ── Tabs: underline style ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 0 !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #b3b3b3 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 14px 22px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }

/* ── Buttons: Spotify pill ── */
.stButton > button {
    background: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 10px 28px !important;
    letter-spacing: 0.4px !important;
    transition: all 0.18s ease !important;
    width: auto !important;
    min-width: 100px !important;
}
.stButton > button:hover {
    background: #1ed760 !important;
    transform: scale(1.04) !important;
    box-shadow: 0 0 24px rgba(29,185,84,0.45) !important;
}

/* Logout button override */
.btn-logout button {
    background: transparent !important;
    color: #b3b3b3 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    font-size: 0.78rem !important;
    padding: 6px 14px !important;
    border-radius: 500px !important;
    letter-spacing: 0.2px !important;
}
.btn-logout button:hover {
    background: rgba(229,9,20,0.12) !important;
    color: #E50914 !important;
    border-color: rgba(229,9,20,0.35) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Save profile button */
.btn-save button {
    background: rgba(29,185,84,0.12) !important;
    color: #1DB954 !important;
    border: 1px solid rgba(29,185,84,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 8px 20px !important;
}
.btn-save button:hover {
    background: rgba(29,185,84,0.22) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Analyze / chart action buttons */
.btn-action button {
    background: transparent !important;
    color: #1DB954 !important;
    border: 1px solid #1DB954 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}
.btn-action button:hover {
    background: rgba(29,185,84,0.1) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Clear chat button */
.btn-clear button {
    background: transparent !important;
    color: #b3b3b3 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 6px 16px !important;
}
.btn-clear button:hover {
    color: #E50914 !important;
    border-color: rgba(229,9,20,0.3) !important;
    background: rgba(229,9,20,0.08) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input,
.stChatInput textarea, .stPasswordInput input {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    direction: rtl !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stPasswordInput input:focus, .stChatInput textarea:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.18) !important;
    background: #222222 !important;
}
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* ── Alerts — forced RTL ── */
.stAlert, [data-testid="stAlert"],
[data-baseweb="notification"],
div[class*="alert"], div[class*="Alert"] {
    direction: rtl !important;
    text-align: right !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.stAlert > div, .stAlert p, .stAlert span { direction: rtl !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    margin-bottom: 10px !important;
    direction: rtl !important;
    padding: 14px 18px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    margin-left: 8% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(29,185,84,0.07) !important;
    border: 1px solid rgba(29,185,84,0.18) !important;
    margin-right: 4% !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    direction: rtl !important;
    text-align: right !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: border-color 0.25s !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(29,185,84,0.3) !important;
}
[data-testid="stMetricValue"] {
    color: #1DB954 !important;
    font-weight: 800 !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetricLabel"] {
    color: #b3b3b3 !important;
    font-weight: 500 !important;
}

/* ── Forms (login/signup glass card) ── */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 28px 24px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #b3b3b3 !important;
    direction: rtl !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 0 0 10px 10px !important;
}
.streamlit-expanderHeader *, .streamlit-expanderContent * { direction: rtl !important; }

/* ── Slider (track stays LTR, labels RTL) ── */
[data-testid="stSlider"] [data-baseweb="slider"] { direction: ltr !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #1DB954 !important; }

/* ── DataTable ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Divider / Spinner / Progress ── */
hr { border-color: rgba(255,255,255,0.07) !important; }
.stSpinner > div { border-top-color: #1DB954 !important; }
.stProgress > div > div > div { background: #1DB954 !important; }

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    direction: rtl !important;
    background: rgba(29,185,84,0.06) !important;
    border: 1px solid rgba(29,185,84,0.18) !important;
    border-radius: 12px !important;
}

/* ── Verified advisor badge ── */
.verified-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(29,185,84,0.14), rgba(29,185,84,0.07));
    border: 1px solid rgba(29,185,84,0.32);
    border-radius: 500px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #1DB954;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
}

/* ── Stock Media Cards ── */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
    gap: 16px;
    margin-top: 20px;
    direction: rtl;
}
.stock-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
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
    transition: opacity 0.25s;
}
.stock-card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: rgba(29,185,84,0.35);
    box-shadow: 0 16px 48px rgba(29,185,84,0.12), 0 4px 16px rgba(0,0,0,0.5);
    background: rgba(29,185,84,0.06);
}
.stock-card:hover::before { opacity: 1; }
.stock-card.card-avoid:hover {
    border-color: rgba(229,9,20,0.35);
    box-shadow: 0 16px 48px rgba(229,9,20,0.1), 0 4px 16px rgba(0,0,0,0.5);
    background: rgba(229,9,20,0.04);
}
.stock-card.card-avoid::before { background: linear-gradient(90deg, #E50914, transparent); }
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 14px;
}
.card-ticker {
    font-size: 1.5rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1;
}
.card-rank {
    font-size: 0.7rem;
    color: #535353;
    font-weight: 500;
    margin-top: 3px;
}
.card-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 500px;
    letter-spacing: 0.2px;
    white-space: nowrap;
}
.badge-buy   { background: rgba(29,185,84,0.18);  color: #1DB954; border: 1px solid rgba(29,185,84,0.4); }
.badge-wait  { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
.badge-avoid { background: rgba(229,9,20,0.18);   color: #E50914; border: 1px solid rgba(229,9,20,0.4); }
.card-price {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 2px;
    letter-spacing: -1px;
    line-height: 1;
}
.card-target {
    font-size: 0.78rem;
    color: #535353;
    margin-bottom: 16px;
}
.card-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin-bottom: 14px;
}
.card-metrics { margin-bottom: 14px; }
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
}
.metric-label { font-size: 0.78rem; color: #727272; }
.metric-value { font-size: 0.85rem; font-weight: 600; color: #ffffff; }
.val-green { color: #1DB954 !important; }
.val-red   { color: #E50914 !important; }
.val-amber { color: #fbbf24 !important; }
.card-footer {
    font-size: 0.78rem;
    color: #535353;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 4px;
}

/* Champion (top pick) card */
.champion-card {
    background: linear-gradient(135deg, rgba(29,185,84,0.12), rgba(29,185,84,0.04));
    border: 1px solid rgba(29,185,84,0.4);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    direction: rtl;
    text-align: right;
    position: relative;
    overflow: hidden;
}
.champion-card::after {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(29,185,84,0.08), transparent 70%);
    pointer-events: none;
}
.champion-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #1DB954;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.champion-ticker-name {
    font-size: 2.4rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -1px;
    margin-bottom: 4px;
}
.champion-sub { font-size: 0.9rem; color: #b3b3b3; }

/* ── Loading animation ── */
@keyframes scan-bar {
    0%   { left: -60%; }
    100% { left: 110%; }
}
@keyframes pulse-text {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.market-loader { text-align: center; padding: 16px 0; direction: rtl; }
.loader-bar {
    height: 2px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    margin: 14px 0;
    position: relative;
}
.loader-bar::after {
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, #1DB954, transparent);
    animation: scan-bar 1.3s linear infinite;
    border-radius: 2px;
}
.loader-text {
    color: #1DB954;
    font-size: 0.85rem;
    font-weight: 600;
    animation: pulse-text 1.4s ease-in-out infinite;
    direction: rtl;
}

/* ── Login screen gradient overlay ── */
.login-gradient {
    background: radial-gradient(ellipse 90% 55% at 50% 0%,
        rgba(29,185,84,0.1) 0%, transparent 65%) !important;
    min-height: 80vh;
}
.login-logo {
    text-align: center;
    padding: 64px 0 48px;
    direction: rtl;
}
.login-logo h1 {
    color: #1DB954 !important;
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    margin: 0 !important;
    letter-spacing: -1.5px !important;
}
.login-logo p {
    color: rgba(255,255,255,0.38) !important;
    font-size: 1.05rem !important;
    margin-top: 10px !important;
    font-weight: 400 !important;
}
.login-footer {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 0.72rem;
    margin-top: 44px;
    direction: rtl;
}

/* ── Sidebar nav section header ── */
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 700;
    color: #535353;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 0 8px 6px;
    direction: rtl;
}

/* ── Server status pill ── */
.server-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 500px;
    font-size: 0.72rem;
    font-weight: 600;
    direction: rtl;
}
.server-pill.online  { background: rgba(29,185,84,0.12); color: #1DB954; border: 1px solid rgba(29,185,84,0.25); }
.server-pill.offline { background: rgba(255,255,255,0.05); color: #727272; border: 1px solid rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN / SIGNUP SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def _render_login_screen() -> None:
    # Gradient overlay just for this page
    st.markdown("""
    <style>
    section.main > div:first-child {
        background: radial-gradient(ellipse 100% 55% at 50% -5%,
            rgba(29,185,84,0.1) 0%, transparent 65%) !important;
    }
    </style>
    <div class="login-logo">
        <div style="font-size:3.5rem; margin-bottom:14px;">📈</div>
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
                name     = st.text_input("👤 שם משתמש")
                password = st.text_input("🔑 סיסמה", type="password")
                submitted = st.form_submit_button("התחבר →")

            if submitted:
                if not name or not password:
                    st.error("יש למלא שם משתמש וסיסמה")
                else:
                    profile = authenticate_user(name, password)
                    if profile:
                        st.session_state.logged_in    = True
                        st.session_state.user_id      = profile["id"]
                        st.session_state.user_name    = profile["name"]
                        st.session_state.user_profile = profile
                        st.session_state.agent        = InvestmentAgent()
                        st.rerun()
                    else:
                        st.error("שם משתמש או סיסמה שגויים — נסה שוב")

        # ── Signup ─────────────────────────────────────────────────────────────
        with tab_signup:
            with st.form("signup_form"):
                new_name     = st.text_input("👤 שם משתמש (ייחודי)")
                new_pass     = st.text_input("🔑 סיסמה", type="password")
                new_pass2    = st.text_input("🔑 אימות סיסמה", type="password")
                new_budget   = st.number_input(
                    "💰 תקציב ($)", min_value=100, max_value=10_000_000, value=10_000, step=500
                )
                risk_opts    = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
                new_risk_lbl = st.selectbox("⚖️ רמת סיכון", list(risk_opts.keys()), index=1)
                new_duration = st.slider("📅 אופק השקעה (חודשים)", 1, 60, 12)
                reg_submitted = st.form_submit_button("הרשמה →")

            if reg_submitted:
                if not new_name or not new_pass:
                    st.error("יש למלא שם משתמש וסיסמה")
                elif new_pass != new_pass2:
                    st.error("הסיסמאות אינן תואמות — נסה שוב")
                else:
                    try:
                        create_user(
                            name=new_name,
                            password=new_pass,
                            budget=float(new_budget),
                            risk_tolerance=risk_opts[new_risk_lbl],
                            duration_months=int(new_duration),
                        )
                        st.success(f"✅ ברוך הבא, {new_name}! עבור ללשונית 'התחברות'")
                    except ValueError as exc:
                        st.error(str(exc))

    st.markdown(
        '<div class="login-footer">⚠️ המידע לצורכי מידע בלבד ואינו ייעוץ פיננסי מוסמך</div>',
        unsafe_allow_html=True,
    )


# ── Auth guard ─────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    _render_login_screen()
    st.stop()

if st.session_state.agent is None:
    st.session_state.agent = InvestmentAgent()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Spotify-style
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(
        "<div style='padding:20px 8px 8px; direction:rtl;'>"
        "<span style='font-size:1.3rem; font-weight:900; color:#1DB954; letter-spacing:-0.5px;'>📈 Pro-Investor</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # User + logout
    _server_ok = st.session_state.backend.ping()
    pill_class = "online" if _server_ok else "offline"
    pill_text  = "שרת מחובר" if _server_ok else "מצב עצמאי"
    pill_dot   = "🟢" if _server_ok else "⚪"

    st.markdown(
        f"<div style='direction:rtl; margin-bottom:6px;'>"
        f"<div style='font-weight:700; color:#ffffff; font-size:1rem;'>שלום, {st.session_state.user_name} 👋</div>"
        f"<span class='server-pill {pill_class}'>{pill_dot} {pill_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="btn-logout">', unsafe_allow_html=True)
    if st.button("🚪 התנתק", key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Investor profile section
    st.markdown('<div class="sidebar-section">💼 פרופיל משקיע</div>', unsafe_allow_html=True)

    _prof = st.session_state.user_profile
    budget = st.number_input(
        "תקציב ($)",
        min_value=100, max_value=10_000_000,
        value=int(_prof.get("budget", 10_000)),
        step=500, format="%d",
    )
    risk_map   = {"נמוך 🟢": "low", "בינוני 🟡": "medium", "גבוה 🔴": "high"}
    _risk_idx  = {"low": 0, "medium": 1, "high": 2}.get(_prof.get("risk_tolerance", "medium"), 1)
    risk_label = st.selectbox("רמת סיכון", list(risk_map.keys()), index=_risk_idx)
    risk       = risk_map[risk_label]
    duration   = st.slider("אופק (חודשים)", 1, 60, int(_prof.get("duration_months", 12)))

    st.markdown('<div class="btn-save">', unsafe_allow_html=True)
    if st.button("💾 שמור פרופיל", key="save_profile"):
        try:
            update_user_profile(st.session_state.user_id, float(budget), risk, int(duration))
            st.session_state.user_profile.update({
                "budget": float(budget), "risk_tolerance": risk, "duration_months": int(duration)
            })
            st.success("✅ נשמר")
        except Exception as exc:
            st.error(f"שגיאה: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Live tickers
    st.markdown('<div class="sidebar-section">📡 שוק בזמן אמת</div>', unsafe_allow_html=True)
    with st.spinner(""):
        try:
            prices = get_multiple_prices(["SPY", "AAPL", "NVDA", "BTC-USD"])
            for sym, price in prices.items():
                if price:
                    st.metric(sym, f"${price:,.2f}")
        except Exception:
            st.caption("נתוני שוק זמינים בצ'אט")

    st.divider()
    st.markdown(
        "<div style='direction:rtl; font-size:0.7rem; color:#3d3d3d; line-height:1.8;'>"
        "🤖 LLaMA-3.3-70b · Groq<br>"
        "🗄️ ChromaDB · Sentence-Transformers<br>"
        "🔍 DuckDuckGo Search"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='direction:rtl; font-size:2.2rem; font-weight:900; color:#ffffff; margin-bottom:4px;'>"
    "📈 Pro-Investor</h1>"
    "<p style='direction:rtl; color:#535353; font-size:0.95rem; margin-top:0;'>"
    "סוכן השקעות AI אוטונומי — מחפש ברשת, מנתח ומדרג בזמן אמת</p>",
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

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="📈"):
                st.markdown(
                    '<div class="verified-badge">📈 Pro-Investor AI &nbsp;·&nbsp; יועץ מאומת</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])
                log = msg.get("tool_log", [])
                if log:
                    with st.expander(f"🔍 תהליך מחקר הסוכן ({len(log)} צעדים)", expanded=False):
                        for step in log:
                            st.markdown(f"{step['icon']} **{step['label']}** — {step['summary']}")

    user_input = st.chat_input("שאל את הסוכן... (לדוגמה: 'נתח NVDA' או 'מה המגמה ב-BTC?')")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        profile = {
            "budget":          budget,
            "risk_tolerance":  risk,
            "duration_months": duration,
            "user_name":       st.session_state.user_name,
        }

        with st.chat_message("assistant", avatar="📈"):
            # Custom loading animation
            loader_ph = st.empty()
            loader_ph.markdown("""
            <div class="market-loader">
                <div class="loader-bar"></div>
                <div class="loader-text">מחפש בשווקים עבורך...</div>
            </div>
            """, unsafe_allow_html=True)

            status = st.status("⟳ הסוכן חוקר...", expanded=False)

            try:
                answer, tool_log = st.session_state.agent.run(user_input, profile)
                loader_ph.empty()

                status.update(
                    label=f"✅ ניתוח הושלם — {len(tool_log)} כלים הופעלו",
                    state="complete",
                    expanded=False,
                )

                st.markdown(
                    '<div class="verified-badge">📈 Pro-Investor AI &nbsp;·&nbsp; יועץ מאומת</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                if tool_log:
                    with st.expander(f"🔍 תהליך מחקר הסוכן ({len(tool_log)} צעדים)", expanded=False):
                        for step in tool_log:
                            st.markdown(f"{step['icon']} **{step['label']}** — {step['summary']}")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "tool_log": tool_log}
                )

                # Persist to DB
                try:
                    for step in tool_log:
                        if step.get("tool") == "get_investment_score" and step.get("data"):
                            d = step["data"]
                            save_recommendation(
                                ticker=d.get("ticker", ""),
                                verdict=d.get("human", {}).get("verdict_raw", "WAIT"),
                                price=float(d.get("current_price", 0)),
                                score_data=d,
                                ai_response=answer,
                                user_id=st.session_state.user_id,
                            )
                            break
                except Exception:
                    pass

            except Exception as exc:
                loader_ph.empty()
                status.update(label="❌ שגיאה", state="error")
                err = f"אני מצטער, נתקלתי בשגיאה טכנית: {exc}"
                st.error(err)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": err, "tool_log": []}
                )

    if st.session_state.chat_history:
        st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
        if st.button("🗑️ נקה שיחה"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Analysis (Media Cards)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown(
        "<h3 style='direction:rtl; color:#ffffff; margin-bottom:20px;'>📊 ניתוח וציון ניירות ערך</h3>",
        unsafe_allow_html=True,
    )

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        tickers_raw = st.text_input(
            "טיקרים (מופרדים בפסיק):", value="AAPL, MSFT, NVDA, SPY, BTC-USD"
        )
    with col_btn:
        st.write("")
        st.markdown('<div class="btn-action">', unsafe_allow_html=True)
        run_analysis = st.button("🔍 נתח", key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    if run_analysis and tickers_raw:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        results = []
        bar = st.progress(0, text="מנתח...")

        for i, sym in enumerate(tickers):
            try:
                results.append(score_ticker(sym, risk, duration))
            except Exception as e:
                st.warning(f"⚠️ {sym}: {e}")
            bar.progress((i + 1) / len(tickers), text=f"מחשב ציון ל-{sym}...")

        bar.empty()

        if results:
            results.sort(key=lambda x: x["investment_score"], reverse=True)

            TREND_MAP = {"bullish": "📈 מגמה עולה", "bearish": "📉 מגמה יורדת", "neutral": "➡️ ניטרלי"}
            VERDICT   = lambda s: (
                ("buy",  "✅ קנייה") if s >= 0.28 else
                ("avoid","❌ הימנע") if s < 0.05 else
                ("wait", "⚠️ המתן")
            )

            # Champion card
            best = results[0]
            bv_cls, bv_txt = VERDICT(best["investment_score"])
            st.markdown(f"""
            <div class="champion-card">
                <div class="champion-label">🥇 המלצה מובילה</div>
                <div class="champion-ticker-name">{best['ticker']}</div>
                <div class="champion-sub">
                    <span class="card-badge badge-{bv_cls}" style="margin-left:10px;">{bv_txt}</span>
                    &nbsp; ${best['current_price']:.2f} &nbsp;→&nbsp; <span style="color:#1DB954;">${best['target_price']:.2f}</span>
                    &nbsp;·&nbsp; {TREND_MAP.get(best['trend'], best['trend'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Build card grid
            cards_html = '<div class="cards-grid">'
            for rank, r in enumerate(results):
                ret_pct    = r["expected_return"] * 100
                vc, vtxt   = VERDICT(r["investment_score"])
                ret_cls    = "val-green" if ret_pct > 0 else "val-red"
                rsi_cls    = "val-green" if r["rsi"] < 45 else ("val-red" if r["rsi"] > 65 else "")
                avoid_cls  = "card-avoid" if vc == "avoid" else ""
                trend_text = TREND_MAP.get(r["trend"], r["trend"])
                prob_pct   = r["probability_profit"] * 100

                cards_html += f"""
                <div class="stock-card {avoid_cls}">
                    <div class="card-header">
                        <div>
                            <div class="card-ticker">{r['ticker']}</div>
                            <div class="card-rank">#{rank+1} מדורג</div>
                        </div>
                        <span class="card-badge badge-{vc}">{vtxt}</span>
                    </div>
                    <div class="card-price">${r['current_price']:.2f}</div>
                    <div class="card-target">יעד: ${r['target_price']:.2f}</div>
                    <div class="card-divider"></div>
                    <div class="card-metrics">
                        <div class="metric-row">
                            <span class="metric-label">תשואה צפויה</span>
                            <span class="metric-value {ret_cls}">{ret_pct:+.1f}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">הסתברות רווח</span>
                            <span class="metric-value">{prob_pct:.0f}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">RSI</span>
                            <span class="metric-value {rsi_cls}">{r['rsi']:.1f}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">תנודתיות</span>
                            <span class="metric-value">{r['volatility']*100:.1f}%</span>
                        </div>
                    </div>
                    <div class="card-footer">{trend_text}</div>
                </div>
                """

            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # Math formula expander
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
        "<h3 style='direction:rtl; color:#ffffff; margin-bottom:20px;'>📈 גרף נרות יפני</h3>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_sym = st.text_input("טיקר:", value="SPY", key="chart_sym")
    with c2:
        period_opts = {"שבוע": "5d", "חודש": "1mo", "3 חודשים": "3mo", "6 חודשים": "6mo", "שנה": "1y"}
        period_lbl  = st.selectbox("תקופה:", list(period_opts.keys()), index=2)

    st.markdown('<div class="btn-action">', unsafe_allow_html=True)
    show_chart = st.button("📊 הצג גרף", key="chart_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if show_chart:
        with st.spinner("טוען נתונים..."):
            try:
                data = get_historical_data(chart_sym.upper(), period_opts[period_lbl])

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data["Open"], high=data["High"],
                    low=data["Low"],   close=data["Close"],
                    increasing_line_color="#1DB954",
                    decreasing_line_color="#E50914",
                    name=chart_sym.upper(),
                ))

                ma20 = data["Close"].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=ma20,
                    name="ממוצע נע 20",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ))

                fig.update_layout(
                    title=dict(text=f"{chart_sym.upper()} — {period_lbl}", x=0.5, font=dict(color="#ffffff")),
                    yaxis_title="מחיר ($)",
                    xaxis_title="",
                    template="plotly_dark",
                    paper_bgcolor="#000000",
                    plot_bgcolor="#0a0a0a",
                    xaxis_rangeslider_visible=False,
                    height=520,
                    legend=dict(orientation="h", y=1.08),
                    font=dict(family="Heebo", color="#b3b3b3"),
                )
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)")
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)")
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
        "<h3 style='direction:rtl; color:#ffffff; margin-bottom:20px;'>📋 היסטוריית המלצות</h3>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="btn-action">', unsafe_allow_html=True)
    if st.button("🔄 רענן", key="refresh_history"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

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
            st.info("📭 אין המלצות שמורות עדיין — שאל את הסוכן על מניה בלשונית ייעוץ AI!")

    except Exception as exc:
        st.error(f"❌ שגיאת בסיס נתונים: {exc}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#2a2a2a; font-size:0.72rem; direction:rtl;'>"
    "⚠️ המידע לצורכי מידע בלבד ואינו ייעוץ פיננסי מוסמך. "
    "השקעות כרוכות בסיכון אובדן הון. | Pro-Investor AI © 2026</p>",
    unsafe_allow_html=True,
)
