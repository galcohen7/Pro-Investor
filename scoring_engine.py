"""
Mathematical scoring engine.

Core formula:
    Score = (Probability_Profit × Expected_Return) / Risk_Factor

Each component is derived from technical indicators and the user's risk profile.

Also provides score_to_human_readable() — converts raw decimals into
beginner-friendly Hebrew labels so the UI never exposes numbers like 0.55.
"""

from technical_indicators import get_all_indicators

# Maps risk-tolerance labels (Hebrew + English) to multipliers and growth targets
RISK_TOLERANCE_CONFIG = {
    "נמוך": {"multiplier": 1.5, "monthly_growth": 0.005},
    "בינוני": {"multiplier": 1.0, "monthly_growth": 0.010},
    "גבוה": {"multiplier": 0.5, "monthly_growth": 0.018},
    "low": {"multiplier": 1.5, "monthly_growth": 0.005},
    "medium": {"multiplier": 1.0, "monthly_growth": 0.010},
    "high": {"multiplier": 0.5, "monthly_growth": 0.018},
}


def calculate_probability_profit(rsi: float, macd_histogram: float, trend: str) -> float:
    """
    Estimates P(profit) in [0.05, 0.95] from three technical signals.

    Signal logic:
      RSI < 30  → oversold  → +0.20 (strong buy signal)
      RSI > 70  → overbought → -0.20 (strong sell signal)
      MACD histogram > 0 → bullish momentum → +0.15
      Trend bullish/bearish → ±0.10
    """
    prob = 0.50  # neutral baseline

    if rsi < 30:
        prob += 0.20
    elif rsi < 45:
        prob += 0.10
    elif rsi > 70:
        prob -= 0.20
    elif rsi > 55:
        prob -= 0.10

    prob += 0.15 if macd_histogram > 0 else -0.15

    trend_delta = {"bullish": 0.10, "neutral": 0.0, "bearish": -0.10}
    prob += trend_delta.get(trend, 0.0)

    return max(0.05, min(0.95, prob))


def estimate_target_price(
    current_price: float,
    bollinger_upper: float,
    trend: str,
    duration_months: int,
    risk_tolerance: str,
) -> float:
    """
    Projects a target price using Bollinger upper band as ceiling
    and a monthly growth rate scaled by risk tolerance and duration.
    """
    config = RISK_TOLERANCE_CONFIG.get(risk_tolerance, RISK_TOLERANCE_CONFIG["medium"])
    monthly_growth = config["monthly_growth"]

    if trend == "bullish":
        base_target = current_price + (bollinger_upper - current_price) * 0.8
    elif trend == "bearish":
        base_target = current_price * 0.95
        monthly_growth *= 0.5
    else:
        base_target = current_price * 1.03

    duration_factor = (1 + monthly_growth) ** duration_months
    return max(base_target * duration_factor, current_price * 0.80)


def calculate_expected_return(current_price: float, target_price: float) -> float:
    """Expected return as a decimal fraction (e.g. 0.15 → 15%)."""
    if current_price <= 0:
        return 0.0
    return (target_price - current_price) / current_price


def calculate_risk_factor(volatility: float, risk_tolerance: str) -> float:
    """
    Risk Factor = annualized volatility × tolerance multiplier.
    High tolerance → smaller multiplier → lower penalty → higher score.
    """
    config = RISK_TOLERANCE_CONFIG.get(risk_tolerance, RISK_TOLERANCE_CONFIG["medium"])
    return max(0.01, volatility * config["multiplier"])


def calculate_investment_score(
    prob_profit: float, expected_return: float, risk_factor: float
) -> float:
    """
    Score = (Probability_Profit × Expected_Return) / Risk_Factor

    Higher score = more attractive investment given the user's risk profile.
    Returns 0 for negative or zero expected return.
    """
    if risk_factor <= 0 or expected_return <= 0:
        return 0.0
    return (prob_profit * expected_return) / risk_factor


def score_to_human_readable(score_data: dict) -> dict:
    """
    Converts raw numerical scores into beginner-friendly Hebrew labels.
    NEVER expose raw decimals (e.g. 0.55) to the end user through this layer.

    Returns a dict with:
      score_label   — descriptive quality rating
      risk_label    — plain-language risk level
      potential     — growth potential with intuitive percentage
      return_str    — expected return as readable string
      verdict       — Hebrew buy/wait/avoid verdict with emoji
      verdict_raw   — machine-readable: BUY / WAIT / AVOID
      trend_he      — Hebrew trend description
      summary_line  — one-sentence Hebrew summary for the AI to cite
    """
    score  = score_data.get("investment_score", 0.0)
    prob   = score_data.get("probability_profit", 0.5)
    ret    = score_data.get("expected_return", 0.0)
    risk   = score_data.get("risk_factor", 0.3)
    trend  = score_data.get("trend", "neutral")
    ticker = score_data.get("ticker", "")

    # ── Investment score label ────────────────────────────────────────────────
    if score >= 0.50:    score_label = "🌟 מצוין"
    elif score >= 0.30:  score_label = "✅ טוב"
    elif score >= 0.15:  score_label = "🟡 בינוני"
    elif score >= 0.05:  score_label = "🟠 חלש"
    else:                score_label = "🔴 לא מומלץ"

    # ── Risk label ────────────────────────────────────────────────────────────
    if risk < 0.12:    risk_label = "סיכון נמוך מאוד 🟢"
    elif risk < 0.22:  risk_label = "סיכון נמוך 🟢"
    elif risk < 0.33:  risk_label = "סיכון בינוני 🟡"
    elif risk < 0.48:  risk_label = "סיכון גבוה 🟠"
    else:              risk_label = "סיכון גבוה מאוד 🔴"

    # ── Growth potential ──────────────────────────────────────────────────────
    prob_pct = round(prob * 100)
    if prob_pct >= 70:    potential = f"פוטנציאל צמיחה גבוה מאוד ({prob_pct}%)"
    elif prob_pct >= 60:  potential = f"פוטנציאל צמיחה גבוה ({prob_pct}%)"
    elif prob_pct >= 50:  potential = f"פוטנציאל צמיחה סביר ({prob_pct}%)"
    elif prob_pct >= 40:  potential = f"פוטנציאל צמיחה נמוך ({prob_pct}%)"
    else:                 potential = f"פוטנציאל צמיחה נמוך מאוד ({prob_pct}%)"

    # ── Expected return string ────────────────────────────────────────────────
    ret_pct = round(ret * 100, 1)
    if ret_pct > 0:
        return_str = f"תשואה צפויה: +{ret_pct}% ב-{score_data.get('duration_months', '')} חודשים" \
                     if score_data.get("duration_months") else f"תשואה צפויה: +{ret_pct}%"
    elif ret_pct < 0:
        return_str = f"ירידה צפויה: {ret_pct}%"
    else:
        return_str = "תשואה צפויה: ניטרלית"

    # ── Trend (Hebrew) ────────────────────────────────────────────────────────
    trend_map = {
        "bullish": "מגמה עולה 📈",
        "bearish": "מגמה יורדת 📉",
        "neutral": "מגמה ניטרלית ➡️",
    }
    trend_he = trend_map.get(trend, trend)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if score >= 0.28 and prob >= 0.58 and trend in ("bullish", "neutral"):
        verdict     = "✅ קנייה"
        verdict_raw = "BUY"
    elif score >= 0.12 or prob >= 0.50:
        verdict     = "⚠️ המתן"
        verdict_raw = "WAIT"
    else:
        verdict     = "❌ לא כרגע"
        verdict_raw = "AVOID"

    # ── One-line summary for AI context ──────────────────────────────────────
    summary_line = (
        f"{ticker}: {score_label} | {risk_label} | {potential} | {trend_he} → {verdict}"
    )

    return {
        "score_label":   score_label,
        "risk_label":    risk_label,
        "potential":     potential,
        "return_str":    return_str,
        "verdict":       verdict,
        "verdict_raw":   verdict_raw,
        "trend_he":      trend_he,
        "summary_line":  summary_line,
    }


def score_ticker(ticker_symbol: str, risk_tolerance: str, duration_months: int) -> dict:
    """
    End-to-end scoring pipeline for a single ticker.
    Fetches indicators, computes all sub-scores, and returns a full breakdown.
    """
    indicators = get_all_indicators(ticker_symbol)

    current_price = indicators["current_price"]
    target_price = estimate_target_price(
        current_price,
        indicators["bollinger_upper"],
        indicators["trend"],
        duration_months,
        risk_tolerance,
    )

    prob_profit = calculate_probability_profit(
        indicators["rsi"],
        indicators["macd_histogram"],
        indicators["trend"],
    )
    expected_return = calculate_expected_return(current_price, target_price)
    risk_factor = calculate_risk_factor(indicators["volatility"], risk_tolerance)
    score = calculate_investment_score(prob_profit, expected_return, risk_factor)

    return {
        "ticker": ticker_symbol,
        "current_price": current_price,
        "target_price": round(target_price, 2),
        "probability_profit": round(prob_profit, 3),
        "expected_return": round(expected_return, 4),
        "risk_factor": round(risk_factor, 4),
        "investment_score": round(score, 6),
        "trend": indicators["trend"],
        "rsi": round(indicators["rsi"], 2),
        "volatility": round(indicators["volatility"], 4),
        "indicators": indicators,
    }
