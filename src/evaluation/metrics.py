"""
Unified Evaluation Metrics and Visualizations for Sentiment Classification.

Supports:
- Standard multi-class classification metrics (Accuracy, Balanced Acc, Macro/Weighted F1, Precision, Recall)
- Per-class metric breakdowns (BULLISH, NEUTRAL, BEARISH)
- Confusion matrix generation and publication-quality visualization
- Metrics logging and tabular comparison
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.utils.common import RESULTS_DIR, get_logger

logger = get_logger(__name__)

SENTIMENT_CLASSES = ["BULLISH", "NEUTRAL", "BEARISH"]


def compute_sentiment_metrics(
    y_true: Union[List[str], np.ndarray, pd.Series],
    y_pred: Union[List[str], np.ndarray, pd.Series],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for financial sentiment.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class ordering (default: ["BULLISH", "NEUTRAL", "BEARISH"]).
        
    Returns:
        Dict of metrics.
    """
    if labels is None:
        labels = SENTIMENT_CLASSES

    CANONICAL_MAP = {
        "POSITIVE": "BULLISH",
        "NEGATIVE": "BEARISH",
        "BULLISH": "BULLISH",
        "BEARISH": "BEARISH",
        "NEUTRAL": "NEUTRAL",
    }
    # Normalize inputs
    y_true = [CANONICAL_MAP.get(str(y).strip().upper(), str(y).strip().upper()) for y in y_true]
    y_pred = [CANONICAL_MAP.get(str(y).strip().upper(), str(y).strip().upper()) for y in y_pred]

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_dict = {}
    for i, label in enumerate(labels):
        support = int(np.sum(np.array(y_true) == label))
        per_class_dict[label] = {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": support,
        }

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(balanced_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "precision": round(float(macro_precision), 4),
        "recall": round(float(macro_recall), 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }


def plot_confusion_matrix(
    y_true: Union[List[str], np.ndarray, pd.Series],
    y_pred: Union[List[str], np.ndarray, pd.Series],
    model_name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    normalize: bool = False,
    title: Optional[str] = None,
) -> Path:
    """
    Generate and save a publication-quality confusion matrix plot.
    """
    display_title = title or model_name or "Model Confusion Matrix"
    if labels is None:
        labels = SENTIMENT_CLASSES

    y_true = [str(y).strip().upper() for y in y_true]
    y_pred = [str(y).strip().upper() for y in y_pred]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Try using seaborn if available, otherwise pure matplotlib
    try:
        import seaborn as sns
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
    except ImportError:
        cax = ax.matshow(cm_display, cmap="Blues")
        fig.colorbar(cax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = f"{cm_display[i, j]:.2%}" if normalize else f"{cm_display[i, j]}"
                ax.text(j, i, val, ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_title(f"Confusion Matrix — {display_title}", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path is None:
        safe_name = display_title.lower().replace(" ", "_").replace("/", "_")
        save_path = RESULTS_DIR / "figures" / f"confusion_matrix_{safe_name}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to {save_path}")
    return save_path


def save_metrics(metrics: Dict[str, Any], filepath: Path):
    """Save metrics dictionary to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {filepath}")
