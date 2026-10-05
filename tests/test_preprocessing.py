"""
Unit tests for data preprocessing and ticker detection.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import clean_text, combine_title_and_body, preprocess_dataframe, create_splits
from src.data.ticker_detection import detect_tickers, extract_cashtags, STOP_TICKERS


def test_clean_text_html_and_urls():
    raw = "Check this out &amp; profit: https://example.com/gainz **massive gain**"
    cleaned = clean_text(raw)
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "https://" not in cleaned
    assert "**" not in cleaned
    assert "massive gain" in cleaned


def test_clean_text_emoji_preservation():
    raw = "Tesla to the moon 🚀 diamond hands 💎🙌"
    cleaned = clean_text(raw)
    assert "EMOJI_ROCKET_BULLISH" in cleaned
    assert "EMOJI_DIAMOND_HANDS" in cleaned


def test_extract_cashtags():
    text = "Buying $AAPL and $TSLA calls, ignoring $FOR and $IT"
    cashtags = extract_cashtags(text)
    assert "AAPL" in cashtags
    assert "TSLA" in cashtags
    assert "FOR" not in cashtags  # In STOP_TICKERS


def test_detect_tickers_cashtag_priority():
    text = "Huge news for $NVDA earnings next week"
    res = detect_tickers(text)
    assert res["primary_ticker"] == "NVDA"
    assert res["confidence"] >= 0.9
    assert res["detection_method"] == "cashtag"


def test_detect_tickers_company_name():
    text = "Microsoft cloud revenue grew tremendously last quarter"
    res = detect_tickers(text)
    assert res["primary_ticker"] == "MSFT"
    assert res["primary_company"] == "Microsoft"
    assert res["detection_method"] == "company_name"


def test_stop_tickers_filtered():
    text = "I WANT TO BUY A NEW CAR FOR MY DOG"
    res = detect_tickers(text)
    # Words like A, FOR, CAR, DOG, NEW should not trigger ticker detection
    assert res["primary_ticker"] is None or res["primary_ticker"] not in ("A", "FOR", "CAR", "DOG", "NEW")


def test_create_splits_ratio():
    df = pd.DataFrame({
        "text": [f"Sample post text {i} with ticker NVDA" for i in range(100)],
        "sentiment_label": ["BULLISH", "BEARISH", "NEUTRAL", "BULLISH"] * 25,
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC"),
    })
    train, val, test = create_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
