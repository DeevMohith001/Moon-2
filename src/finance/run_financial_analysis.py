"""
Master Pipeline for Financial Market Analysis (Phase 7).

Integrates:
- Stock price downloads (yfinance)
- Temporal market hour alignment
- Sentiment aggregation
- Econometric statistics (F1, F2)
- Directional predictive models (F3, F4, F5)
- Event spike analysis (F6)
- Engagement weighting analysis (F7)
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import (
    DATA_PROC,
    RESULTS_DIR,
    load_config,
    log_experiment,
    get_logger,
)
from src.data.ingestion import assert_real_data, DataUnavailableError
from src.finance.prices import fetch_historical_prices, compute_forward_returns
from src.finance.alignment import align_timestamps_to_trading_days
from src.finance.aggregation import aggregate_daily_sentiment
from src.finance.returns import build_sentiment_return_dataset
from src.finance.statistics import (
    compute_correlations,
    run_ols_regressions,
    evaluate_predictive_models,
    run_event_analysis,
    plot_sentiment_return_scatter,
)
from src.evaluation.metrics import save_metrics

logger = get_logger(__name__)


def run_full_financial_pipeline():
    """
    Run end-to-end financial analysis pipeline.

    IMPORTANT:
    - Requires REAL Reddit data and REAL market data.
    - Hard-fails on synthetic data.
    - Uses strict timestamp alignment to prevent look-ahead leakage.
    - Results are CORRELATIONAL, NOT causal. Do not interpret as trading signals.
    """
    cfg = load_config()
    fin_cfg = cfg.get("finance", {})
    horizons = fin_cfg.get("return_horizons", [1, 3, 5])
    min_posts = fin_cfg.get("min_posts_per_day", 1)

    # 1. Load processed posts
    proc_file = DATA_PROC / "processed_reddit.parquet"
    if not proc_file.exists():
        logger.error(f"Processed Reddit dataset not found: {proc_file}")
        return

    reddit_df = pd.read_parquet(proc_file)

    # PROVENANCE CHECK: hard-fail on synthetic data
    assert_real_data(reddit_df)

    logger.info(f"Loaded processed Reddit data: {len(reddit_df)} posts (verified REAL)")

    # Focus on top liquid tickers with high post counts to avoid thousands of slow HTTP calls
    target_candidates = ["AAPL", "TSLA", "MSFT", "NVDA", "GME", "GOOGL", "SPY", "AMZN", "META", "AMD"]
    tickers = [t for t in target_candidates if t in reddit_df["ticker"].values]
    if not tickers:
        tickers = reddit_df["ticker"].value_counts().head(10).index.tolist()

    reddit_df = reddit_df[reddit_df["ticker"].isin(tickers)].copy()
    logger.info(f"Target tickers for financial analysis ({len(tickers)}): {tickers} ({len(reddit_df)} posts)")

    # Score posts if sentiment_score is missing
    if "sentiment_score" not in reddit_df.columns or reddit_df["sentiment_score"].isna().all():
        logger.info("Scoring real Reddit posts using VADER baseline in single pass...")
        from src.baselines.vader import VADERBaseline
        vader = VADERBaseline()
        texts_trunc = [str(t)[:1000] for t in reddit_df["text"]]
        labels, scores = vader.predict(texts_trunc)
        reddit_df["sentiment_label"] = labels
        reddit_df["sentiment_score"] = scores

    # Determine date range (bounded to active market years)
    ts = pd.to_datetime(reddit_df["timestamp"], utc=True)
    start_date = max(pd.to_datetime("2020-01-01", utc=True), ts.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = min(pd.to_datetime("2025-07-20", utc=True), ts.max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    # 2. Fetch prices and compute forward returns
    prices_raw = fetch_historical_prices(tickers, start_date=start_date, end_date=end_date)
    prices_fwd = compute_forward_returns(prices_raw, horizons=horizons)

    # 3. Align timestamps to market calendar
    valid_dates = sorted(prices_fwd["date"].unique().tolist())
    aligned_reddit = align_timestamps_to_trading_days(reddit_df, valid_trading_dates=valid_dates)
    logger.info(f"Aligned {len(aligned_reddit)} posts to active US trading days.")

    # 4. Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(aligned_reddit, min_posts=min_posts)

    # 5. Build merged dataset
    merged_df = build_sentiment_return_dataset(daily_sentiment, prices_fwd)
    merged_path = DATA_PROC / "sentiment_returns_merged.parquet"
    merged_df.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged sentiment-return dataset to {merged_path}")

    # 6. Statistical Analysis
    # F1: Correlations (unweighted vs weighted)
    unweighted_corr = compute_correlations(merged_df, horizons=horizons, sentiment_col="mean_sentiment")
    weighted_corr = compute_correlations(merged_df, horizons=horizons, sentiment_col="weighted_sentiment")

    # F2: OLS Regressions
    ols_res = run_ols_regressions(merged_df, horizons=horizons, sentiment_col="mean_sentiment")

    # F3, F4, F5: Directional Predictions
    pred_res = evaluate_predictive_models(merged_df, horizon=1)

    # F6: Event Spike Analysis
    event_res = run_event_analysis(merged_df, sentiment_col="mean_sentiment")

    # Save comprehensive results
    summary = {
        "correlations_unweighted_F1": unweighted_corr,
        "correlations_weighted_F7": weighted_corr,
        "ols_regressions_F2": ols_res,
        "predictive_models_F3_F4_F5": pred_res,
        "event_analysis_F6": event_res,
        "total_joint_observations": len(merged_df),
        "total_tickers": merged_df["ticker"].nunique(),
    }

    metrics_path = RESULTS_DIR / "metrics" / "financial_analysis_results.json"
    save_metrics(summary, metrics_path)

    # Generate scatter plot
    plot_path = plot_sentiment_return_scatter(merged_df)
    logger.info(f"Saved scatter plot to {plot_path}")

    # Track experiments
    log_experiment("F1", {"analysis": "Unweighted Correlation", "corr_1d": unweighted_corr.get("1d", {}).get("pearson_r")})
    log_experiment("F2", {"analysis": "OLS Regression 1d", "beta_1": ols_res.get("1d", {}).get("beta_1"), "r2": ols_res.get("1d", {}).get("r_squared")})
    if pred_res:
        log_experiment("F3", {"analysis": "Market Only Model", "acc": pred_res.get("F3_market_only", {}).get("accuracy")})
        log_experiment("F4", {"analysis": "Sentiment Only Model", "acc": pred_res.get("F4_sentiment_only", {}).get("accuracy")})
        log_experiment("F5", {"analysis": "Combined Model", "acc": pred_res.get("F5_combined", {}).get("accuracy")})
    log_experiment("F6", {"analysis": "Event Spike Spread 1d", "spread_1d": event_res.get("spread_1d")})
    log_experiment("F7", {"analysis": "Weighted Correlation 1d", "corr_1d": weighted_corr.get("1d", {}).get("pearson_r")})

    print("\n" + "=" * 65)
    print("FINANCIAL MARKET ANALYSIS SUMMARY (Phase 7)")
    print("=" * 65)
    print(f"Total Joint Observations: {len(merged_df)} across {merged_df['ticker'].nunique()} tickers")
    print("\nCorrelation with Forward Returns (Pearson r):")
    for h in horizons:
        r = unweighted_corr.get(f"{h}d", {}).get("pearson_r")
        p = unweighted_corr.get(f"{h}d", {}).get("pearson_p_value")
        print(f"  {h}-Day Horizon: r = {r:+.4f} (p-value: {p:.4f})")
    print(f"\nEngagement-Weighted 1-Day Horizon: r = {weighted_corr.get('1d', {}).get('pearson_r'):+.4f}")
    print(f"\n1-Day Return Spread (Bullish Spikes vs Bearish Spikes): {event_res.get('spread_1d', 0)*100:+.2f}%")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    run_full_financial_pipeline()
