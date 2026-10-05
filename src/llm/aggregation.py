"""
Multi-Path Aggregation, Majority Voting, and Soft Score Computation.

Implements Deng et al. (2023) Section 3.1 & 3.2:
- 8 predictions -> Majority Vote for direct LLM evaluation
- Explicit, auditable tie-breaking strategy
- Continuous Soft Agreement Score calculation:
  positive_count, neutral_count, negative_count, dominant_label, dominant_count, agreement_score
- Continuous sentiment target: teacher_soft_score = (positive_count - negative_count) / num_paths in [-1.0, 1.0]
- Generates weak_labels.csv for student model distillation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.common import DATA_PROC, RESULTS_DIR, OUTPUTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Numeric mapping for continuous score
SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def aggregate_post_paths(
    paths_df: pd.DataFrame,
    tie_breaking_strategy: str = "neutral_default",
) -> Dict[str, Any]:
    """
    Aggregate K reasoning paths for a single post.
    Computes majority label, tie status, class counts, agreement score, and soft continuous score.
    """
    labels = paths_df["label"].tolist()
    total_paths = len(labels)

    pos_count = labels.count("positive")
    neu_count = labels.count("neutral")
    neg_count = labels.count("negative")

    counts = {"positive": pos_count, "neutral": neu_count, "negative": neg_count}

    # Find maximum count and candidate winners
    max_count = max(counts.values())
    top_labels = [l for l, c in counts.items() if c == max_count]

    is_tie = len(top_labels) > 1

    if not is_tie:
        dominant_label = top_labels[0]
    else:
        if tie_breaking_strategy == "neutral_default":
            # Ties represent teacher uncertainty -> fallback to neutral
            dominant_label = "neutral"
        elif tie_breaking_strategy == "highest_confidence":
            # Pick candidate with highest mean confidence
            candidate_confs = {}
            for l in top_labels:
                c_vals = paths_df[paths_df["label"] == l]["confidence"].tolist()
                candidate_confs[l] = float(np.mean(c_vals)) if c_vals else 0.0
            dominant_label = max(candidate_confs, key=candidate_confs.get)
        else:
            dominant_label = top_labels[0]

    agreement_score = max_count / total_paths if total_paths > 0 else 0.0

    # Continuous soft score in [-1.0, 1.0]
    # Positives add +1.0, Negatives add -1.0, Neutrals add 0.0
    teacher_soft_score = (pos_count - neg_count) / total_paths if total_paths > 0 else 0.0

    post_id = str(paths_df["post_id"].iloc[0])

    # Extract first non-empty reasoning summary for inspection
    reasoning_sample = ""
    for r in paths_df["reasoning_summary"].dropna():
        if len(str(r).strip()) > 10:
            reasoning_sample = str(r).strip()
            break

    return {
        "post_id": post_id,
        "positive_count": pos_count,
        "neutral_count": neu_count,
        "negative_count": neg_count,
        "total_paths": total_paths,
        "dominant_label": dominant_label,
        "dominant_count": max_count,
        "is_tie": is_tie,
        "agreement_score": round(agreement_score, 4),
        "teacher_soft_score": round(teacher_soft_score, 4),
        "reasoning_sample": reasoning_sample,
    }


def aggregate_paths_dataframe(
    paths_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
    tie_breaking_strategy: str = "neutral_default",
) -> pd.DataFrame:
    """
    Aggregate all multi-path predictions across posts and merge with original metadata.
    """
    logger.info(f"Aggregating {len(paths_df)} reasoning paths across unique posts...")
    grouped = paths_df.groupby("post_id", sort=False)

    aggregated_records = []
    for post_id, group in grouped:
        rec = aggregate_post_paths(group, tie_breaking_strategy=tie_breaking_strategy)
        aggregated_records.append(rec)

    agg_df = pd.DataFrame(aggregated_records)

    # Merge with original post text and metadata if provided
    if original_df is not None:
        orig_copy = original_df.copy()
        if "post_id" in orig_copy.columns:
            orig_copy["post_id"] = orig_copy["post_id"].astype(str)
            agg_df["post_id"] = agg_df["post_id"].astype(str)
            merged = pd.merge(agg_df, orig_copy, on="post_id", how="left")
            agg_df = merged

    logger.info(
        f"Aggregated {len(agg_df)} posts. Mean agreement score: {agg_df['agreement_score'].mean():.3f}"
    )
    return agg_df


def save_weak_labels(
    agg_df: pd.DataFrame,
    filename: str = "weak_labels.csv",
) -> Path:
    """
    Save the aggregated weak labels dataset to data/processed/, results/, and outputs/metrics/.
    """
    out_paths = [
        DATA_PROC / filename,
        OUTPUTS_DIR / "metrics" / filename,
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(p, index=False)
        logger.info(f"Saved weak labels dataset to {p}")

    return out_paths[0]
