"""
Market Hour Alignment and Trading Calendar Mapping.

Aligns Reddit post creation timestamps (UTC) with US equity market hours:
- Regular trading hours: 09:30 - 16:00 US/Eastern.
- Posts after 16:00 ET align to next trading session (t+1).
- Posts on weekends / holidays map to the subsequent trading day.
- Completely prevents lookahead bias.
"""

from typing import List, Set
import pandas as pd
import numpy as np


def align_timestamps_to_trading_days(
    reddit_df: pd.DataFrame,
    valid_trading_dates: List[object],
    timestamp_col: str = "timestamp",
    market_close_hour: int = 16,
) -> pd.DataFrame:
    """
    Map each post's timestamp to the appropriate effective trading date.
    
    Args:
        reddit_df: DataFrame containing Reddit posts.
        valid_trading_dates: List or array of valid trading day dates (sorted).
        timestamp_col: Name of datetime column.
        market_close_hour: Hour cutoff (16:00 ET).
        
    Returns:
        DataFrame with added 'trading_date' column.
    """
    df = reddit_df.copy()

    # Convert to UTC datetime if not already
    ts = pd.to_datetime(df[timestamp_col], utc=True)

    # Convert to US Eastern Time
    ts_et = ts.dt.tz_convert("US/Eastern")

    # If hour >= 16 (4 PM) or weekend, shift calendar date forward
    raw_dates = ts_et.dt.date
    after_hours_mask = ts_et.dt.hour >= market_close_hour

    effective_dates = []
    sorted_trading_dates = sorted(list(set(valid_trading_dates)))
    trading_dates_arr = np.array(sorted_trading_dates)

    for d, after_h in zip(raw_dates, after_hours_mask):
        target_d = d + pd.Timedelta(days=1) if after_h else d
        # Find next valid trading date >= target_d
        idx = np.searchsorted(trading_dates_arr, target_d)
        if idx < len(trading_dates_arr):
            effective_dates.append(trading_dates_arr[idx])
        else:
            effective_dates.append(None)

    df["trading_date"] = effective_dates
    df = df[df["trading_date"].notna()].copy()
    return df
