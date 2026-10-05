"""
Financial Alignment and Merging of Daily Sentiment with Equity Returns.

Merges daily aggregated sentiment features with forward asset returns,
lagged market momentum, and trading volume.
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import get_logger

logger = get_logger(__name__)


def build_sentiment_return_dataset(
    daily_sentiment_df: pd.DataFrame,
    price_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ticker daily sentiment signals with price forward returns.
    
    Args:
        daily_sentiment_df: columns ['ticker', 'trading_date', 'mean_sentiment', ...]
        price_returns_df: columns ['ticker', 'date', 'fwd_return_1d', 'fwd_return_3d', ...]
        
    Returns:
        Merged analysis DataFrame.
    """
    s_df = daily_sentiment_df.copy()
    p_df = price_returns_df.copy()

    # Align date representations to string or date objects
    s_df["date_str"] = pd.to_datetime(s_df["trading_date"]).dt.strftime("%Y-%m-%d")
    p_df["date_str"] = pd.to_datetime(p_df["date"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(
        s_df,
        p_df,
        on=["ticker", "date_str"],
        how="inner",
        suffixes=("", "_price"),
    )

    # Clean redundant columns
    drop_cols = [c for c in ["date_str", "date"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    logger.info(f"Merged sentiment with market returns: {len(merged)} joint observations across {merged['ticker'].nunique()} tickers.")
    return merged
