"""
Unit and integration tests for data integrity, provenance guards, and split leakage.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.ingestion import assert_real_data, DataUnavailableError, _standardize_hf_reddit, load_fiqa_dataset
from src.data.preprocessing import create_splits, preprocess_dataframe


def test_assert_real_data_rejects_synthetic():
    """Verify assert_real_data hard-fails on synthetic markers."""
    # Tagged data_source
    df_syn = pd.DataFrame({"text": ["test post"], "data_source": ["SYNTHETIC"]})
    with pytest.raises(DataUnavailableError):
        assert_real_data(df_syn)

    # Tagged source
    df_src = pd.DataFrame({"text": ["test post"], "source": ["synthetic_reddit_v1"]})
    with pytest.raises(DataUnavailableError):
        assert_real_data(df_src)


def test_assert_real_data_accepts_real():
    """Verify assert_real_data passes on real verified data."""
    df_real = pd.DataFrame({
        "text": ["AAPL quarterly earnings beat"],
        "source": ["emilpartow/reddit_finance_posts_apple-tesla-microsoft"],
        "data_source": ["REAL"],
    })
    # Should not raise
    assert_real_data(df_real)


def test_create_splits_zero_leakage_and_mutually_exclusive():
    """Verify train, val, and test splits have 0 post and text overlap."""
    posts = [
        {"post_id": f"p_{i}", "text": f"Unique financial post content number {i}", "sentiment_label": "BULLISH" if i % 2 == 0 else "BEARISH"}
        for i in range(100)
    ]
    df = pd.DataFrame(posts)

    train_df, val_df, test_df = create_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    # Check partition sizes
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check zero overlap in IDs
    train_ids = set(train_df["post_id"])
    val_ids = set(val_df["post_id"])
    test_ids = set(test_df["post_id"])
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Check zero overlap in text
    train_txt = set(train_df["text"])
    val_txt = set(val_df["text"])
    test_txt = set(test_df["text"])
    assert len(train_txt.intersection(val_txt)) == 0
    assert len(train_txt.intersection(test_txt)) == 0
    assert len(val_txt.intersection(test_txt)) == 0


def test_fiqa_loader_protocol_and_subtasks():
    """Verify FiQA loader separates subtasks, removes score==0, and maps binary labels."""
    pytest.importorskip("datasets")

    news_splits = load_fiqa_dataset(subtask="news")
    assert "test" in news_splits
    test_news = news_splits["test"]
    assert len(test_news) > 0
    assert (test_news["continuous_score"] == 0).sum() == 0
    assert set(test_news["sentiment_label"].unique()).issubset({"BULLISH", "BEARISH"})
    assert (test_news["data_source"] == "REAL").all()

    post_splits = load_fiqa_dataset(subtask="post")
    assert "test" in post_splits
    test_post = post_splits["test"]
    assert len(test_post) > 0
    assert (test_post["continuous_score"] == 0).sum() == 0
    assert set(test_post["sentiment_label"].unique()).issubset({"BULLISH", "BEARISH"})


def test_standardize_hf_reddit_schema():
    """Verify standardization produces required columns and types."""
    dummy_hf = pd.DataFrame({
        "id": ["123"],
        "title": ["Tesla stock rise"],
        "text": ["Autonomous driving is here"],
        "created_utc": [1620000000.0],
        "score": [150],
        "num_comments": [45],
        "subreddit": ["stocks"],
    })
    standardized = _standardize_hf_reddit(dummy_hf, source="hf_test")
    assert "post_id" in standardized.columns
    assert "timestamp" in standardized.columns
    assert "body" in standardized.columns
    assert "data_source" in standardized.columns
    assert standardized["data_source"].iloc[0] == "REAL"
