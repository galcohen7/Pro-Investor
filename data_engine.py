import yfinance as yf
import pandas as pd


def get_live_price(ticker_symbol: str) -> float:
    """Fetches the latest closing price for a given financial asset ticker."""
    asset = yf.Ticker(ticker_symbol)
    data = asset.history(period="1d")
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker_symbol}")
    return float(data["Close"].iloc[-1])


def get_historical_data(ticker_symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given ticker.
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    asset = yf.Ticker(ticker_symbol)
    data = asset.history(period=period)
    if data.empty:
        raise ValueError(f"No historical data for ticker: {ticker_symbol}")
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
