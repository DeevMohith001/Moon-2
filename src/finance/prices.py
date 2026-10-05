"""
Historical Stock Price Acquisition and Forward Return Computation.

Uses yfinance to download daily price history for equities, caches to parquet,
and computes forward returns for specified trading day horizons (1d, 3d, 5d).
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import yfinance as yf

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import DATA_CACHE, DATA_PROC, get_logger

logger = get_logger(__name__)


def fetch_historical_prices(
    tickers: List[str],
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-05",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for specified tickers.
    
    Returns:
        DataFrame with columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    cache_dir = DATA_CACHE / "finance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"prices_{'_'.join(sorted(tickers)[:5])}_{len(tickers)}.parquet"

    if cache and cache_file.exists():
        logger.info(f"Loading cached price data from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Fetching historical prices for {len(tickers)} tickers ({start_date} to {end_date})...")
    all_dfs = []

    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            df = t.history(start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                logger.warning(f"No price data returned for ticker {sym}")
                continue

            df = df.reset_index()
            # Normalize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["ticker"] = sym
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
            all_dfs.append(df)
            logger.info(f"  Fetched {sym}: {len(df)} trading days")
        except Exception as e:
            logger.warning(f"Failed to fetch {sym}: {e}")

    if not all_dfs:
        raise RuntimeError("No historical price data could be fetched.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    if cache:
        combined.to_parquet(cache_file, index=False)
        logger.info(f"Cached price data saved to {cache_file}")

    return combined


def compute_forward_returns(
    price_df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5],
) -> pd.DataFrame:
    """
    Compute forward returns for each ticker over trading horizons.
    Forward return at day t for horizon h:
      R_{t, t+h} = (Close_{t+h} - Close_t) / Close_t
    Also computes 1-day lagged return for market baseline feature.
    """
    df = price_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    # 1-day lagged return (prior day market momentum)
    df["lag_return_1d"] = df.groupby("ticker")["adj_close"].pct_change(1)

    for h in horizons:
        # Forward return: shift backward so at day t we have return looking ahead h days
        df[f"fwd_return_{h}d"] = df.groupby("ticker")["adj_close"].pct_change(h).shift(-h)
        # Directional binary target (1 if positive forward return, else 0)
        df[f"fwd_direction_{h}d"] = (df[f"fwd_return_{h}d"] > 0).astype(int)

    return df
