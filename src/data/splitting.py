"""
Dataset splitting strategies: chronological (temporal) and stratified splits.
Ensures zero data leakage between train, validation, and test partitions.
"""

from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically based on timestamps.
    Prevents temporal lookahead leakage in market analysis.
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    logger.info(
        f"Chronological split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def split_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    label_col: str = "sentiment_label",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset with stratification to preserve class distributions.
    """
    test_ratio = 1.0 - (train_ratio + val_ratio)
    stratify = df[label_col] if label_col in df.columns and df[label_col].notna().all() else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify,
        random_state=random_seed,
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)
    temp_stratify = temp_df[label_col] if stratify is not None else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction,
        stratify=temp_stratify,
        random_state=random_seed,
    )

    logger.info(
        f"Stratified split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_dataset_splits(
    df: pd.DataFrame,
    strategy: str = "stratified",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Unified entry point for generating train/val/test splits.
    """
    if strategy == "chronological":
        return split_chronological(df, train_ratio=train_ratio, val_ratio=val_ratio)
    return split_stratified(
        df, train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed
    )
