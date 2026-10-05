"""
Daily Sentiment Aggregation and Signal Construction (Section 21).

For each ticker and trading day, computes:
- positive_ratio: bullish_count / volume_of_posts
- negative_ratio: bearish_count / volume_of_posts
- neutral_ratio: neutral_count / volume_of_posts
- net_sentiment: positive_ratio - negative_ratio
- volume_of_posts: total Reddit posts for ticker/day
- average_post_score: average Reddit upvote score
- sentiment_change: day-over-day delta in net_sentiment
- rolling_sentiment: 1-day, 3-day, and 7-day rolling window means
- engagement_weighted_sentiment: log1p(upvotes + comments) weighted score
"""

from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_daily_sentiment(
    aligned_df: pd.DataFrame,
    min_posts: int = 1,
) -> pd.DataFrame:
    """
    Aggregate Reddit sentiment scores into daily ticker-level financial signals.
    """
    df = aligned_df.copy()

    # Ensure scalar sentiment score exists in [-1.0, 1.0]
    if "sentiment_score" not in df.columns:
        score_map = {
            "BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0,
            "positive": 1.0, "negative": -1.0, "neutral": 0.0,
        }
        df["sentiment_score"] = df["sentiment_label"].map(score_map).fillna(0.0)

    score_col = df["score"] if "score" in df.columns else 1
    comments_col = df["num_comments"] if "num_comments" in df.columns else 0
    df["engagement_weight"] = np.log1p(np.maximum(0, score_col) + np.maximum(0, comments_col))
    df["engagement_weight"] = df["engagement_weight"].replace(0, 1.0)
    df["weighted_score_prod"] = df["sentiment_score"] * df["engagement_weight"]

    grouped = df.groupby(["ticker", "trading_date"])

    agg_df = grouped.agg(
        volume_of_posts=("sentiment_score", "count"),
        average_post_score=("score", "mean") if "score" in df.columns else ("sentiment_score", lambda x: 1.0),
        mean_sentiment=("sentiment_score", "mean"),
        median_sentiment=("sentiment_score", "median"),
        std_sentiment=("sentiment_score", "std"),
        sum_weighted_prod=("weighted_score_prod", "sum"),
        sum_weights=("engagement_weight", "sum"),
        bullish_count=("sentiment_score", lambda s: (s > 0).sum()),
        bearish_count=("sentiment_score", lambda s: (s < 0).sum()),
        neutral_count=("sentiment_score", lambda s: (s == 0).sum()),
    ).reset_index()

    # Aliases
    agg_df["post_count"] = agg_df["volume_of_posts"]
    agg_df["std_sentiment"] = agg_df["std_sentiment"].fillna(0.0)

    # Core ratios (Section 21)
    agg_df["positive_ratio"] = agg_df["bullish_count"] / agg_df["volume_of_posts"]
    agg_df["negative_ratio"] = agg_df["bearish_count"] / agg_df["volume_of_posts"]
    agg_df["neutral_ratio"] = agg_df["neutral_count"] / agg_df["volume_of_posts"]
    agg_df["net_sentiment"] = agg_df["positive_ratio"] - agg_df["negative_ratio"]
    agg_df["bull_bear_ratio"] = agg_df["net_sentiment"]

    # Engagement weighted
    agg_df["weighted_sentiment"] = agg_df["sum_weighted_prod"] / np.maximum(agg_df["sum_weights"], 1e-6)
    agg_df = agg_df.drop(columns=["sum_weighted_prod", "sum_weights"])

    # Filter min posts
    agg_df = agg_df[agg_df["volume_of_posts"] >= min_posts].copy()

    # Calculate rolling signals per ticker
    agg_df["trading_date"] = pd.to_datetime(agg_df["trading_date"])
    agg_df = agg_df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)

    rolling_dfs = []
    for ticker, group in agg_df.groupby("ticker"):
        g = group.copy().sort_values("trading_date")
        g["sentiment_change"] = g["net_sentiment"].diff().fillna(0.0)
        g["rolling_sentiment_1d"] = g["net_sentiment"]
        g["rolling_sentiment_3d"] = g["net_sentiment"].rolling(window=3, min_periods=1).mean()
        g["rolling_sentiment_7d"] = g["net_sentiment"].rolling(window=7, min_periods=1).mean()
        rolling_dfs.append(g)

    if rolling_dfs:
        agg_df = pd.concat(rolling_dfs, ignore_index=True)
    else:
        agg_df["sentiment_change"] = 0.0
        agg_df["rolling_sentiment_1d"] = agg_df["net_sentiment"]
        agg_df["rolling_sentiment_3d"] = agg_df["net_sentiment"]
        agg_df["rolling_sentiment_7d"] = agg_df["net_sentiment"]

    logger.info(
        f"Aggregated daily signals for {len(agg_df)} ticker-date rows across {agg_df['ticker'].nunique()} tickers."
    )
    return agg_df
