"""
Categorized Error Analysis and Model Disagreement Diagnostics (Phase 8).

Analyzes misclassifications across:
- Sarcasm & Memes (e.g., 🤡, "literally cannot go tits up", "guh", "paper hands")
- Financial Jargon & Complex Derivatives (e.g., IV crush, gamma squeeze, max pain)
- Mixed / Conflicting Sentiment (e.g., "beat earnings but guidance is terrible")
- Ambiguous or Multi-Ticker Mentions
- Implicit Market Stance
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import RESULTS_DIR, get_logger

logger = get_logger(__name__)

ERROR_PATTERNS = {
    "SARCASM_MEMES": [
        r"\bclown\b", r"🤡", r"guh", r"tits up", r"paper hands", r"diamond hands",
        r"bagholder", r"to the moon", r"rocket", r"stonks", r"loss porn",
    ],
    "FINANCIAL_JARGON": [
        r"\biv crush\b", r"\bgamma squeeze\b", r"\bmax pain\b", r"\btheta\b",
        r"\bdelta\b", r"\b10-k\b", r"\b10-q\b", r"\bearnings call\b", r"\bguidance cut\b",
        r"\bdilution\b", r"\brestructuring\b",
    ],
    "MIXED_SENTIMENT": [
        r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\byet\b", r"\bon the other hand\b",
        r"\bdespite\b", r"\blong term.*short term\b", r"\bshort term.*long term\b",
    ],
    "AMBIGUOUS_TICKER": [
        r"\band\b.*\b(calls|puts)\b", r"\bvs\b", r"\bcompare\b", r"\bor\b",
    ],
}


def classify_error_category(text: str) -> str:
    """Classify the primary suspected reason for model classification difficulty."""
    lower = text.lower()
    for category, patterns in ERROR_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower):
                return category
    return "IMPLICIT_OR_SUBTLE_SENTIMENT"


def run_error_diagnostics(
    pred_file: Path,
    model_name: str,
) -> pd.DataFrame:
    """
    Diagnose errors in a predictions parquet file.
    """
    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return pd.DataFrame()

    df = pd.read_parquet(pred_file)
    if "is_correct" not in df.columns:
        df["is_correct"] = df["sentiment_label"] == df["pred_label"]

    errors_df = df[~df["is_correct"]].copy()
    if errors_df.empty:
        logger.info(f"Zero errors found for {model_name}.")
        return pd.DataFrame()

    errors_df["error_category"] = errors_df["text"].apply(classify_error_category)
    errors_df["model_name"] = model_name

    logger.info(f"Analyzed {len(errors_df)} errors for {model_name}. Breakdown:\n{errors_df['error_category'].value_counts()}")
    return errors_df


def analyze_all_model_errors() -> pd.DataFrame:
    """
    Run diagnostic error analysis across all available model prediction outputs.
    """
    pred_dir = RESULTS_DIR / "predictions"
    pred_files = [
        (pred_dir / "vader_predictions.parquet", "VADER"),
        (pred_dir / "logreg_predictions.parquet", "TF-IDF + LogReg"),
        (pred_dir / "svm_predictions.parquet", "TF-IDF + Linear SVM"),
        (pred_dir / "prosusai_finbert_predictions.parquet", "FinBERT (ProsusAI)"),
        (pred_dir / "llm_l1_predictions.parquet", "LLM Zero-Shot"),
        (pred_dir / "llm_l5_predictions.parquet", "LLM CoT"),
        (pred_dir / "student_regression_predictions.parquet", "Distilled Student"),
    ]

    all_errors = []
    for fpath, name in pred_files:
        if fpath.exists():
            e_df = run_error_diagnostics(fpath, name)
            if not e_df.empty:
                all_errors.append(e_df)

    if not all_errors:
        logger.warning("No prediction errors found to compile.")
        return pd.DataFrame()

    combined_errors = pd.concat(all_errors, ignore_index=True)
    out_csv = RESULTS_DIR / "error_analysis.csv"
    combined_errors.to_csv(out_csv, index=False)
    logger.info(f"Saved comprehensive error analysis to {out_csv}")

    # Plot error category breakdown
    plot_error_breakdown(combined_errors)
    return combined_errors


def plot_error_breakdown(errors_df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Plot distribution of error types across models."""
    counts = errors_df.groupby(["model_name", "error_category"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="black", alpha=0.85)

    ax.set_title("Categorized Error Distribution Across Sentiment Models", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Error Count", fontsize=11)
    ax.legend(title="Error Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / "figures" / "error_distribution.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    analyze_all_model_errors()
