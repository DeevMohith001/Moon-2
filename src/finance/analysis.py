"""
Financial Market Analysis Pipeline (Project Extension).

NOTE: This section is an exploratory extension not present in the original paper.
DISCLAIMER: For academic and research purposes only. This analysis does not
constitute financial advice, trading signals, or market recommendations.

Implements:
- Correlation Analysis (Pearson & Spearman across 1d, 3d, 5d horizons)
- Lag Cross-Correlation Analysis (-2 to +2 day horizons)
- High-Attention Event Studies (top percentile volume & sentiment spikes)
- OLS Regression of Forward Returns against Net Sentiment
- Predictive Model Comparison (Market-only vs. Sentiment-only vs. Combined)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.finance.statistics import (
    compute_correlations,
    run_ols_regressions,
    run_event_analysis,
    compare_predictive_models,
)
from src.utils.common import OUTPUTS_DIR, RESULTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "DISCLAIMER: This project is for research and educational purposes only. "
    "Sentiment predictions and market analyses should not be interpreted as financial advice or trading recommendations."
)


def compute_lag_correlations(
    df: pd.DataFrame,
    sentiment_col: str = "net_sentiment",
    return_col: str = "fwd_return_1d",
    max_lags: int = 3,
) -> pd.DataFrame:
    """
    Compute lead/lag correlations between daily sentiment and equity returns.
    Lags < 0: return leads sentiment.
    Lags > 0: sentiment leads return.
    """
    records = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("trading_date").reset_index(drop=True)
        if len(g) < max_lags * 3:
            continue

        for lag in range(-max_lags, max_lags + 1):
            if lag < 0:
                s_shifted = g[sentiment_col].iloc[-lag:]
                r_shifted = g[return_col].iloc[:lag]
            elif lag > 0:
                s_shifted = g[sentiment_col].iloc[:-lag]
                r_shifted = g[return_col].iloc[lag:]
            else:
                s_shifted = g[sentiment_col]
                r_shifted = g[return_col]

            valid = pd.DataFrame({"s": s_shifted.values, "r": r_shifted.values}).dropna()
            if len(valid) >= 10:
                r_val, p_val = stats.pearsonr(valid["s"], valid["r"])
                records.append({
                    "ticker": ticker,
                    "lag_days": lag,
                    "pearson_r": round(float(r_val), 4),
                    "p_value": round(float(p_val), 4),
                    "n_samples": len(valid),
                })

    res_df = pd.DataFrame(records)
    return res_df


def run_full_financial_analysis(
    merged_df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5],
    sentiment_col: str = "net_sentiment",
) -> Dict[str, Any]:
    """
    Execute complete empirical financial analysis and save outputs.
    """
    logger.info("Running empirical financial analysis (Project Extension)...")
    logger.info(f"Analysis notice: {DISCLAIMER}")

    # 1. Correlation Analysis
    corr_results = compute_correlations(merged_df, horizons=horizons, sentiment_col=sentiment_col)

    # 2. OLS Regressions
    ols_results = run_ols_regressions(merged_df, horizons=horizons, sentiment_col=sentiment_col)

    # 3. Event Analysis
    event_results = run_event_analysis(merged_df, horizons=horizons, percentile=0.90)

    # 4. Lag Analysis
    lag_df = compute_lag_correlations(merged_df, sentiment_col=sentiment_col)

    # 5. Predictive Models
    pred_models = compare_predictive_models(merged_df, horizon=1)

    summary = {
        "disclaimer": DISCLAIMER,
        "correlations": corr_results,
        "ols_regressions": ols_results,
        "event_analysis": event_results,
        "predictive_models": pred_models,
        "num_analyzed_trading_days": len(merged_df),
        "covered_tickers": merged_df["ticker"].unique().tolist() if "ticker" in merged_df.columns else [],
    }

    # Save summary
    for target_dir in [OUTPUTS_DIR / "reports", RESULTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(target_dir / "financial_analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info("Completed financial market analysis.")
    return summary
