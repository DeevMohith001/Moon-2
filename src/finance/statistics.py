"""
Statistical and Econometric Analysis of Financial Sentiment:
- Experiment F1: Pearson and Spearman Correlation (1d, 3d, 5d forward returns)
- Experiment F2: OLS Linear Regression (Returns ~ Sentiment)
- Experiment F3: Market-Only Directional Prediction (Baseline)
- Experiment F4: Sentiment-Only Directional Prediction
- Experiment F5: Combined Market + Sentiment Directional Prediction
- Experiment F6: High-Activity Event Analysis (Sentiment Spikes)
- Experiment F7: Engagement-Weighted vs Unweighted Sentiment Comparison
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import RESULTS_DIR, log_experiment, get_logger
from src.evaluation.metrics import save_metrics

logger = get_logger(__name__)


def compute_correlations(
    df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5],
    sentiment_col: str = "mean_sentiment",
) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman rank correlation between sentiment and forward returns.
    """
    corr_results = {}
    valid_df = df.dropna(subset=[sentiment_col] + [f"fwd_return_{h}d" for h in horizons])

    for h in horizons:
        ret_col = f"fwd_return_{h}d"
        p_corr, p_val = stats.pearsonr(valid_df[sentiment_col], valid_df[ret_col])
        s_corr, s_val = stats.spearmanr(valid_df[sentiment_col], valid_df[ret_col])

        corr_results[f"{h}d"] = {
            "pearson_r": round(float(p_corr), 4),
            "pearson_p_value": round(float(p_val), 4),
            "spearman_rho": round(float(s_corr), 4),
            "spearman_p_value": round(float(s_val), 4),
            "n_samples": len(valid_df),
        }

    return corr_results


def run_ols_regressions(
    df: pd.DataFrame,
    horizons: List[int] = [1, 3, 5],
    sentiment_col: str = "mean_sentiment",
) -> Dict[str, Any]:
    """
    Fit OLS regression: Forward Return = beta_0 + beta_1 * Sentiment + epsilon.
    """
    ols_results = {}
    valid_df = df.dropna(subset=[sentiment_col] + [f"fwd_return_{h}d" for h in horizons])

    for h in horizons:
        ret_col = f"fwd_return_{h}d"
        x = valid_df[sentiment_col].values
        y = valid_df[ret_col].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ols_results[f"{h}d"] = {
            "beta_1": round(float(slope), 5),
            "intercept": round(float(intercept), 5),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 4),
            "std_error": round(float(std_err), 5),
        }

    return ols_results


def evaluate_predictive_models(
    df: pd.DataFrame,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Compare directional prediction models (Experiments F3, F4, F5):
    - Model A (F3): Market momentum only (lag_return_1d)
    - Model B (F4): Sentiment only (mean_sentiment, post_count)
    - Model C (F5): Combined (lag_return_1d + mean_sentiment + post_count)
    """
    target_col = f"fwd_direction_{horizon}d"
    features_all = ["lag_return_1d", "mean_sentiment", "post_count"]

    valid_df = df.dropna(subset=features_all + [target_col]).copy()
    if len(valid_df) < 30:
        logger.warning("Insufficient samples for robust predictive model training.")
        return {}

    # Chronological train/test split (80/20) to prevent lookahead
    split_idx = int(len(valid_df) * 0.8)
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:]

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Check if target has both classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"note": "Single-class distribution in split"}

    model_configs = {
        "F3_market_only": ["lag_return_1d"],
        "F4_sentiment_only": ["mean_sentiment", "post_count"],
        "F5_combined": ["lag_return_1d", "mean_sentiment", "post_count"],
    }

    results = {}
    for exp_id, cols in model_configs.items():
        clf = LogisticRegression(random_state=42)
        clf.fit(train_df[cols], y_train)

        preds = clf.predict(test_df[cols])
        probs = clf.predict_proba(test_df[cols])[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        f1 = f1_score(y_test, preds, zero_division=0)

        results[exp_id] = {
            "features": cols,
            "accuracy": round(float(acc), 4),
            "auc_roc": round(float(auc), 4),
            "f1_score": round(float(f1), 4),
            "train_size": len(train_df),
            "test_size": len(test_df),
        }

    return results


def run_event_analysis(
    df: pd.DataFrame,
    sentiment_col: str = "mean_sentiment",
    threshold_percentile: float = 85.0,
) -> Dict[str, Any]:
    """
    Event analysis (F6): Compare forward returns on high-sentiment vs low-sentiment spike days.
    """
    valid_df = df.dropna(subset=[sentiment_col, "fwd_return_1d", "fwd_return_3d"]).copy()
    if valid_df.empty:
        return {}

    high_thresh = np.percentile(valid_df[sentiment_col], threshold_percentile)
    low_thresh = np.percentile(valid_df[sentiment_col], 100 - threshold_percentile)

    bull_burst = valid_df[valid_df[sentiment_col] >= high_thresh]
    bear_burst = valid_df[valid_df[sentiment_col] <= low_thresh]

    results = {
        "bullish_events": {
            "count": len(bull_burst),
            "mean_fwd_1d": round(float(bull_burst["fwd_return_1d"].mean()), 4),
            "mean_fwd_3d": round(float(bull_burst["fwd_return_3d"].mean()), 4),
        },
        "bearish_events": {
            "count": len(bear_burst),
            "mean_fwd_1d": round(float(bear_burst["fwd_return_1d"].mean()), 4),
            "mean_fwd_3d": round(float(bear_burst["fwd_return_3d"].mean()), 4),
        },
        "spread_1d": round(float(bull_burst["fwd_return_1d"].mean() - bear_burst["fwd_return_1d"].mean()), 4),
        "spread_3d": round(float(bull_burst["fwd_return_3d"].mean() - bear_burst["fwd_return_3d"].mean()), 4),
    }

    return results


def plot_sentiment_return_scatter(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Generate publication scatter plot of daily sentiment vs 1-day forward return."""
    valid_df = df.dropna(subset=["mean_sentiment", "fwd_return_1d"])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    x = valid_df["mean_sentiment"]
    y = valid_df["fwd_return_1d"] * 100  # convert to percentage

    ax.scatter(x, y, alpha=0.6, edgecolors="none", c="#1f77b4", s=40, label="Daily observations")

    # Fit line
    if len(valid_df) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="#d62728", linewidth=2.5, label=f"OLS Trend (slope={m:.3f})")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Reddit Daily Sentiment vs. 1-Day Forward Equity Returns", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Aggregated Daily Sentiment Score (-1.0 = Bearish, +1.0 = Bullish)", fontsize=11)
    ax.set_ylabel("1-Day Forward Return (%)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "figures" / "sentiment_vs_return_scatter.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
