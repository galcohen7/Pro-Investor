"""
Mathematical scoring engine.

Core formula:
    Score = (Probability_Profit × Expected_Return) / Risk_Factor

Each component is derived from technical indicators and the user's risk profile.
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
