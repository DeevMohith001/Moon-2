"""
Unit tests for financial alignment and sentiment aggregation.
"""

import pandas as pd
import numpy as np
from src.finance.alignment import align_timestamps_to_trading_days
from src.finance.aggregation import aggregate_daily_sentiment


def test_trading_day_alignment():
    # Post at 5 PM ET on Friday 2023-01-06 -> should map to Monday 2023-01-09
    df = pd.DataFrame({
        "timestamp": ["2023-01-06 22:00:00+00:00"],  # 5 PM ET
        "text": ["Buying shares"],
        "ticker": ["AAPL"],
        "sentiment_score": [1.0],
    })
    valid_dates = [pd.to_datetime("2023-01-06").date(), pd.to_datetime("2023-01-09").date()]
    aligned = align_timestamps_to_trading_days(df, valid_trading_dates=valid_dates)
    assert len(aligned) == 1
    assert str(aligned.iloc[0]["trading_date"]) == "2023-01-09"


def test_sentiment_aggregation_math():
    df = pd.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "trading_date": ["2023-01-09", "2023-01-09"],
        "sentiment_score": [1.0, -1.0],
        "sentiment_label": ["BULLISH", "BEARISH"],
        "score": [10, 5],
        "num_comments": [2, 1],
    })
    agg = aggregate_daily_sentiment(df, min_posts=1)
    assert len(agg) == 1
    assert agg.iloc[0]["post_count"] == 2
    assert agg.iloc[0]["mean_sentiment"] == 0.0
    assert agg.iloc[0]["bullish_count"] == 1
    assert agg.iloc[0]["bearish_count"] == 1


if __name__ == "__main__":
    test_trading_day_alignment()
    test_sentiment_aggregation_math()
    print("Finance tests passed!")
