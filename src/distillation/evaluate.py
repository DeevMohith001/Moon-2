"""
Evaluation and Threshold Tuning for Distilled Student Models.

Implements Deng et al. (2023) Section 3.2:
- Grid search over threshold theta on validation set to maximize Macro F1
- Threshold mapping: s > theta -> BULLISH, s < -theta -> BEARISH, else NEUTRAL
- Unified evaluation metrics: Accuracy, Macro F1, Weighted F1, MSE, MAE, Confusion Matrix
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.distillation.dataset import TextRegressionDataset, TextClassificationDataset, ID_TO_LABEL
from src.evaluation.metrics import compute_sentiment_metrics, plot_confusion_matrix, save_metrics
from src.utils.common import OUTPUTS_DIR, RESULTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def tune_regression_threshold(
    val_preds: List[float],
    val_true_labels: List[str],
    search_thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Grid search threshold theta in [0.05, 0.40] optimizing validation Macro F1.
    """
    search_thresholds = search_thresholds or [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best_theta = 0.20
    best_f1 = -1.0

    for theta in search_thresholds:
        preds = [
            "BULLISH" if p > theta else ("BEARISH" if p < -theta else "NEUTRAL")
            for p in val_preds
        ]
        m = compute_sentiment_metrics(val_true_labels, preds)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_theta = theta

    logger.info(f"Optimal validation threshold theta: {best_theta:.2f} (Val Macro F1: {best_f1:.4f})")
    return best_theta, best_f1


def evaluate_regression_student(
    model: torch.nn.Module,
    tokenizer: Any,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    batch_size: int = 16,
    theta: Optional[float] = None,
    save_prefix: str = "distillation_regression",
) -> Dict[str, Any]:
    """
    Evaluate regression student model on holdout test set.
    """
    device = torch.device("cpu")
    model.eval()

    text_col = "clean_text" if "clean_text" in test_df.columns else "text"
    true_labels = test_df["sentiment_label"].tolist()

    # If theta not provided, tune on validation set
    if theta is None and val_df is not None:
        v_text_col = "clean_text" if "clean_text" in val_df.columns else "text"
        val_ds = TextRegressionDataset(val_df[v_text_col].tolist(), [0.0] * len(val_df), tokenizer)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                val_preds.extend(out.logits.squeeze(1).cpu().numpy())
        theta, _ = tune_regression_threshold(val_preds, val_df["sentiment_label"].tolist())
    elif theta is None:
        theta = 0.20

    # Evaluate on test set
    test_ds = TextRegressionDataset(test_df[text_col].tolist(), [0.0] * len(test_df), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    test_preds_raw = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds_raw.extend(out.logits.squeeze(1).cpu().numpy())

    discrete_preds = [
        "BULLISH" if p > theta else ("BEARISH" if p < -theta else "NEUTRAL")
        for p in test_preds_raw
    ]

    metrics = compute_sentiment_metrics(true_labels, discrete_preds)
    metrics["optimal_theta"] = theta
    metrics["mean_predicted_score"] = round(float(np.mean(test_preds_raw)), 4)
    metrics["std_predicted_score"] = round(float(np.std(test_preds_raw)), 4)

    # Save artifacts
    for target_dir in [OUTPUTS_DIR, RESULTS_DIR]:
        metrics_file = target_dir / "metrics" / f"{save_prefix}_metrics.json"
        save_metrics(metrics, metrics_file)
        fig_file = target_dir / "figures" / f"confusion_matrix_{save_prefix}.png"
        plot_confusion_matrix(
            true_labels,
            discrete_preds,
            title="Distilled Student (Regression Loss)",
            save_path=fig_file,
        )

    logger.info(
        f"Regression Student Test Evaluation: Accuracy={metrics['accuracy']*100:.2f}%, Macro F1={metrics['macro_f1']:.4f}"
    )
    metrics["predictions"] = discrete_preds
    metrics["raw_scores"] = [float(x) for x in test_preds_raw]
    return metrics


def evaluate_classification_student(
    model: torch.nn.Module,
    tokenizer: Any,
    test_df: pd.DataFrame,
    batch_size: int = 16,
    save_prefix: str = "distillation_classification",
) -> Dict[str, Any]:
    """
    Evaluate classification student baseline model on holdout test set.
    """
    device = torch.device("cpu")
    model.eval()

    text_col = "clean_text" if "clean_text" in test_df.columns else "text"
    true_labels = test_df["sentiment_label"].tolist()

    test_ds = TextClassificationDataset(
        test_df[text_col].tolist(),
        true_labels,
        tokenizer,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    pred_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            for p in preds:
                pred_labels.append(ID_TO_LABEL.get(p, "NEUTRAL"))

    metrics = compute_sentiment_metrics(true_labels, pred_labels)
    metrics["predictions"] = pred_labels

    # Save artifacts
    for target_dir in [OUTPUTS_DIR, RESULTS_DIR]:
        metrics_file = target_dir / "metrics" / f"{save_prefix}_metrics.json"
        save_metrics(metrics, metrics_file)
        fig_file = target_dir / "figures" / f"confusion_matrix_{save_prefix}.png"
        plot_confusion_matrix(
            true_labels,
            pred_labels,
            title="Distilled Student (Classification Baseline)",
            save_path=fig_file,
        )

    logger.info(
        f"Classification Student Test Evaluation: Accuracy={metrics['accuracy']*100:.2f}%, Macro F1={metrics['macro_f1']:.4f}"
    )
    return metrics
