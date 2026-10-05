"""
Data Validation and Quality Assurance for Reddit Financial Sentiment Dataset.

Performs:
- Schema checks and missing value profiling
- Class balance verification
- Ticker coverage and distribution analysis
- Text length and token distribution
- Temporal coverage checks
- Generation of data quality report
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.common import DATA_PROC, REPORTS_DIR, get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, Any]:
    """Check required columns and null counts."""
    missing_cols = [c for c in required_cols if c not in df.columns]
    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    null_pcts = {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns}

    return {
        "all_required_present": len(missing_cols) == 0,
        "missing_columns": missing_cols,
        "null_counts": null_counts,
        "null_percentages": null_pcts,
    }


def analyze_text_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute character and word length statistics."""
    if "text" not in df.columns:
        return {}

    lengths = df["text"].str.len()
    words = df["text"].str.split().str.len()

    return {
        "char_count": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
            "std": round(float(lengths.std()), 1),
        },
        "word_count": {
            "min": int(words.min()),
            "max": int(words.max()),
            "mean": round(float(words.mean()), 1),
            "median": round(float(words.median()), 1),
            "std": round(float(words.std()), 1),
        },
    }


def generate_quality_report(
    df: pd.DataFrame,
    split_name: str = "full",
    save_path: Path = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive dataset quality report.
    """
    logger.info(f"Generating data quality report for split: {split_name} ({len(df)} rows)")

    required = ["text", "timestamp", "ticker"]
    schema_results = validate_schema(df, required)

    report = {
        "split": split_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "schema_validation": schema_results,
        "text_statistics": analyze_text_statistics(df),
    }

    # Sentiment distribution
    if "sentiment_label" in df.columns:
        counts = df["sentiment_label"].value_counts().to_dict()
        normalized = df["sentiment_label"].value_counts(normalize=True).round(4).to_dict()
        report["sentiment_distribution"] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): float(v * 100) for k, v in normalized.items()},
        }

    # Ticker coverage
    if "ticker" in df.columns:
        valid_tickers = df["ticker"].dropna()
        ticker_counts = valid_tickers.value_counts().head(15).to_dict()
        report["ticker_coverage"] = {
            "total_posts_with_ticker": int(len(valid_tickers)),
            "coverage_pct": round(float(len(valid_tickers) / max(len(df), 1) * 100), 2),
            "unique_tickers": int(valid_tickers.nunique()),
            "top_tickers": {str(k): int(v) for k, v in ticker_counts.items()},
        }

    # Subreddit distribution
    if "subreddit" in df.columns:
        sub_counts = df["subreddit"].value_counts().to_dict()
        report["subreddit_distribution"] = {str(k): int(v) for k, v in sub_counts.items()}

    # Temporal coverage
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        report["temporal_coverage"] = {
            "min_date": str(ts.min()),
            "max_date": str(ts.max()),
            "total_days_span": int((ts.max() - ts.min()).days) if ts.notna().any() else 0,
        }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Data quality report saved to {save_path}")

    return report


if __name__ == "__main__":
    proc_file = DATA_PROC / "processed_reddit.parquet"
    if proc_file.exists():
        df = pd.read_parquet(proc_file)
        report = generate_quality_report(
            df,
            split_name="processed_full",
            save_path=REPORTS_DIR / "data_quality_report.json",
        )
        print("\n--- Data Quality Summary ---")
        print(f"Total Rows: {report['total_rows']}")
        print(f"Sentiment Distribution: {report.get('sentiment_distribution', {}).get('counts')}")
        print(f"Ticker Coverage: {report.get('ticker_coverage', {}).get('coverage_pct')}%")
    else:
        print(f"File {proc_file} does not exist yet. Run src/data/preprocessing.py first.")
