import yfinance as yf
import pandas as pd


def get_live_price(ticker_symbol: str) -> float:
    """Fetches the latest closing price for a given financial asset ticker."""
    asset = yf.Ticker(ticker_symbol)
    data = asset.history(period="1d")
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker_symbol}")
    return float(data["Close"].iloc[-1])


# Maps each period to the finest interval yfinance allows for that window.
# yfinance hard limits: 1m→7d, 5m/15m/30m→60d, 1h→730d, 1d→unlimited.
_AUTO_INTERVAL: dict[str, str] = {
    "1d":  "5m",
    "5d":  "15m",
    "1mo": "1h",
    "3mo": "1d",
    "6mo": "1d",
    "1y":  "1d",
    "2y":  "1d",
    "5y":  "1wk",
    "max": "1mo",
}


def get_historical_data(
    ticker_symbol: str,
    period: str = "3mo",
    interval: str | None = None,
) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given ticker.

    If interval is None, the finest interval supported by yfinance for the
    requested period is selected automatically (e.g. 5m for 1d, 15m for 5d).

    period options : 1d 5d 1mo 3mo 6mo 1y 2y 5y max
    interval options: 1m 2m 5m 15m 30m 1h 1d 1wk 1mo
    """
    if interval is None:
        interval = _AUTO_INTERVAL.get(period, "1d")

    asset = yf.Ticker(ticker_symbol)
    data  = asset.history(period=period, interval=interval)
    if data.empty:
        raise ValueError(f"No historical data for {ticker_symbol} (period={period}, interval={interval})")
    return data


def get_asset_info(ticker_symbol: str) -> dict:
    """Fetches fundamental metadata about an asset (sector, market cap, P/E, 52-week range)."""
    asset = yf.Ticker(ticker_symbol)
    info = asset.info
    return {
        "name": info.get("longName", ticker_symbol),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", None),
        "forward_pe": info.get("forwardPE", None),
        "52w_high": info.get("fiftyTwoWeekHigh", None),
        "52w_low": info.get("fiftyTwoWeekLow", None),
        "avg_volume": info.get("averageVolume", 0),
        "dividend_yield": info.get("dividendYield", 0),
    }


def get_multiple_prices(ticker_symbols: list) -> dict:
    """Fetches live prices for multiple tickers; returns None for failed lookups."""
    prices = {}
    for symbol in ticker_symbols:
        try:
            prices[symbol] = get_live_price(symbol)
        except Exception:
            prices[symbol] = None
    return prices
