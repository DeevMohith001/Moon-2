"""
PyTorch Datasets and Consistency Filtering for Knowledge Distillation.

Implements Deng et al. (2023) Section 3.2:
- Consistency filtering across thresholds: 8/8, 7/8, 6/8, 5/8 (default >= 5/8)
- TextRegressionDataset: pairs text with continuous teacher_soft_score in [-1.0, 1.0]
- TextClassificationDataset: pairs text with integer class label (0=negative, 1=neutral, 2=positive)
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so filtering utilities remain importable without PyTorch."""

        pass

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Class label mappings for classification objective
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_DISPLAY = {"negative": "BEARISH", "neutral": "NEUTRAL", "positive": "BULLISH"}


def filter_by_consistency(
    weak_df: pd.DataFrame,
    threshold: int = 5,
    total_paths: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter weakly labeled examples by agreement threshold (e.g. >= 5 out of 8).
    Returns filtered DataFrame and detailed filtering statistics.
    """
    if "dominant_count" not in weak_df.columns:
        raise ValueError("DataFrame must contain 'dominant_count' column for consistency filtering.")

    initial_count = len(weak_df)
    filtered_df = weak_df[weak_df["dominant_count"] >= threshold].copy().reset_index(drop=True)
    retained_count = len(filtered_df)
    retention_ratio = retained_count / initial_count if initial_count > 0 else 0.0

    class_dist = (
        filtered_df["dominant_label"].value_counts(normalize=True).to_dict()
        if retained_count > 0
        else {}
    )

    stats = {
        "threshold": f"{threshold}/{total_paths}",
        "initial_count": initial_count,
        "retained_count": retained_count,
        "retention_count": retained_count,
        "retention_ratio": round(retention_ratio, 4),
        "retention_pct": round(retention_ratio * 100, 2),
        "mean_agreement": (
            round(float(filtered_df["agreement_score"].mean()), 4)
            if retained_count > 0
            else 0.0
        ),
        "class_distribution": class_dist,
    }

    logger.info(
        f"Consistency Filtering [{threshold}/{total_paths}]: Retained {retained_count}/{initial_count} posts ({stats['retention_pct']}%)"
    )
    return filtered_df, stats


class TextRegressionDataset(Dataset):
    """
    PyTorch Dataset for Regression Distillation (Experiment R2).
    Encodes text and pairs with continuous target score in [-1.0, 1.0].
    """

    def __init__(
        self,
        texts: List[str],
        targets: List[float],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to create TextRegressionDataset.")
        self.encodings = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["target"] = self.targets[idx]
        return item


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for Classification Distillation Baseline (Experiment R1).
    Encodes text and pairs with integer class labels: 0=negative, 1=neutral, 2=positive.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[Any],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to create TextClassificationDataset.")
        self.encodings = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        int_labels = []
        for l in labels:
            if isinstance(l, int):
                int_labels.append(l)
            else:
                s = str(l).strip().lower()
                int_labels.append(LABEL_TO_ID.get(s, 1))

        self.labels = torch.tensor(int_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
