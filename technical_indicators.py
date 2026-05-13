import numpy as np
import pandas as pd

from data_engine import get_historical_data


def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    """
    Relative Strength Index (RSI).
    < 30 = oversold (buy signal), > 70 = overbought (sell signal).
    """
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calculate_macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """
    Moving Average Convergence Divergence (MACD).
    Returns (macd_line, signal_line, histogram) as latest scalar values.
    """
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def calculate_bollinger_bands(
    closes: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """
    Bollinger Bands: (upper_band, middle_band, lower_band) as latest scalar values.
    Price near upper band = potentially overbought; near lower = oversold.
    """
    rolling_mean = closes.rolling(window=window).mean()
    rolling_std = closes.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return float(upper_band.iloc[-1]), float(rolling_mean.iloc[-1]), float(lower_band.iloc[-1])


def calculate_volatility(closes: pd.Series, window: int = 20) -> float:
    """Annualized historical volatility using log returns (σ × √252)."""
    log_returns = np.log(closes / closes.shift(1))
    daily_vol = log_returns.rolling(window=window).std().iloc[-1]
    return float(daily_vol * np.sqrt(252))


def determine_trend(closes: pd.Series, short_ma: int = 20, long_ma: int = 50) -> str:
    """
    Trend detection via moving average crossover.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    if len(closes) < long_ma:
        return "neutral"
    ma_short = closes.rolling(window=short_ma).mean().iloc[-1]
    ma_long = closes.rolling(window=long_ma).mean().iloc[-1]
    if ma_short > ma_long * 1.01:
        return "bullish"
    elif ma_short < ma_long * 0.99:
        return "bearish"
    return "neutral"


def get_all_indicators(ticker_symbol: str) -> dict:
    """
    Full technical analysis pipeline for one ticker.
    Fetches 6 months of history and computes RSI, MACD, Bollinger Bands,
    volatility, and trend direction.
    """
    data = get_historical_data(ticker_symbol, period="6mo")
    closes = data["Close"]

    rsi = calculate_rsi(closes)
    macd, signal, histogram = calculate_macd(closes)
    upper_bb, mid_bb, lower_bb = calculate_bollinger_bands(closes)
    volatility = calculate_volatility(closes)
    trend = determine_trend(closes)
    current_price = float(closes.iloc[-1])

    return {
        "ticker": ticker_symbol,
        "current_price": current_price,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "macd_histogram": histogram,
        "bollinger_upper": upper_bb,
        "bollinger_middle": mid_bb,
        "bollinger_lower": lower_bb,
        "volatility": volatility,
        "trend": trend,
    }
