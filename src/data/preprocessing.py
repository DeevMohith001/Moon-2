"""
Data Preprocessing Pipeline for Reddit Financial Sentiment Analysis.

Transforms raw Reddit posts into clean, model-ready format:
- Text normalization (markdown, HTML, URLs, emoji handling)
- Text filtering (minimum length, non-informative posts)
- Ticker detection and resolution
- Temporal normalization (UTC)
- Split generation (train/val/test with strict chronological or stratified split)
"""

import re
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.common import (
    DATA_RAW, DATA_INTERIM, DATA_PROC,
    load_config, get_logger, set_seed
)
from src.data.ticker_detection import detect_tickers

logger = get_logger(__name__)

# Common financial emoji sentiment mapping (sentiment-bearing emojis)
FINANCIAL_EMOJIS = {
    "🚀": " EMOJI_ROCKET_BULLISH ",
    "🌙": " EMOJI_MOON_BULLISH ",
    "💎": " EMOJI_DIAMOND_HANDS ",
    "🙌": " EMOJI_DIAMOND_HANDS ",
    "🐂": " EMOJI_BULL_MARKET ",
    "📈": " EMOJI_CHART_UP ",
    "🔥": " EMOJI_FIRE_HOT ",
    "🟢": " EMOJI_GREEN_UP ",
    "🐻": " EMOJI_BEAR_MARKET ",
    "📉": " EMOJI_CHART_DOWN ",
    "🩸": " EMOJI_BLOODY_RED ",
    "🔴": " EMOJI_RED_DOWN ",
    "🤡": " EMOJI_CLOWN_SARCASTIC ",
    "💀": " EMOJI_SKULL_DEAD ",
    "⚰️": " EMOJI_COFFIN_DEAD ",
    "💩": " EMOJI_POOP_BAD ",
}


def clean_text(text: str, preserve_financial_emojis: bool = True) -> str:
    """
    Clean Reddit post text:
    - Decode HTML entities
    - Remove markdown formatting
    - Replace or strip URLs
    - Preserve key financial emojis by translating to tokens
    - Remove Reddit metadata ([deleted], [removed])
    - Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)

    # Discard Reddit deleted/removed indicators
    if text.strip() in ("[deleted]", "[removed]", "null", "nan"):
        return ""

    # Remove URLs (replace with token or remove)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove markdown formatting: bold, italic, code blocks, headers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)  # Quote blocks

    # Translate key financial emojis into sentiment tokens
    if preserve_financial_emojis:
        for emoji_char, token in FINANCIAL_EMOJIS.items():
            text = text.replace(emoji_char, token)

    # Remove non-ascii characters except for basic punctuation and our translated emoji tokens
    # Note: keep letters, numbers, spaces, and punctuation
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_title_and_body(title: str, body: str) -> str:
    """Combine post title and body with clear delimiter."""
    title_clean = clean_text(title or "")
    body_clean = clean_text(body or "")

    if title_clean and body_clean:
        # Avoid duplicate title if body already starts with title
        if body_clean.startswith(title_clean):
            return body_clean
        return f"{title_clean} -- {body_clean}"
    elif title_clean:
        return title_clean
    elif body_clean:
        return body_clean
    return ""


def preprocess_dataframe(df: pd.DataFrame, min_text_len: int = 15) -> pd.DataFrame:
    """
    Apply full cleaning and enrichment pipeline to DataFrame:
    1. Filter missing titles and bodies
    2. Combine and clean text
    3. Filter short texts (< min_text_len)
    4. Detect tickers and companies
    5. Deduplicate by cleaned text
    6. Normalize timestamps
    """
    logger.info(f"Starting preprocessing on {len(df)} raw records...")
    processed = df.copy()

    # Explicitly ensure data_source is preserved
    if "data_source" not in processed.columns:
        source_val = str(processed.get("source", pd.Series(["unknown"])).iloc[0] if len(processed) > 0 else "")
        processed["data_source"] = "SYNTHETIC" if "synthetic" in source_val.lower() else "REAL"
    processed["text"] = [
        combine_title_and_body(t, b)
        for t, b in zip(processed.get("title", [""] * len(processed)),
                        processed.get("body", [""] * len(processed)))
    ]

    # Filter out empty or too short text
    processed = processed[processed["text"].str.len() >= min_text_len].copy()
    logger.info(f"After length filtering (>= {min_text_len} chars): {len(processed)} records")

    # Deduplicate by text
    initial_len = len(processed)
    processed = processed.drop_duplicates(subset=["text"]).copy()
    logger.info(f"Deduplicated {initial_len - len(processed)} duplicate texts. Remaining: {len(processed)}")

    # Detect tickers and companies
    tickers_info = [detect_tickers(txt) for txt in processed["text"]]
    processed["ticker"] = [t["primary_ticker"] for t in tickers_info]
    processed["company"] = [t["primary_company"] for t in tickers_info]
    processed["all_tickers"] = [t["all_tickers"] for t in tickers_info]
    processed["ticker_confidence"] = [t["confidence"] for t in tickers_info]
    processed["detection_method"] = [t["detection_method"] for t in tickers_info]

    # Ground truth sentiment if present in raw (e.g. from synthetic dev dataset or labeled benchmark)
    if "_sentiment" in processed.columns and "sentiment_label" not in processed.columns:
        processed["sentiment_label"] = processed["_sentiment"]
    elif "sentiment_label" not in processed.columns:
        processed["sentiment_label"] = None

    # Score mapping: BULLISH -> 1.0, BEARISH -> -1.0, NEUTRAL -> 0.0
    sentiment_to_score = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
    if "sentiment_label" in processed.columns and processed["sentiment_label"].notna().any():
        processed["sentiment_score"] = processed["sentiment_label"].map(sentiment_to_score)
    else:
        processed["sentiment_score"] = np.nan

    # Ensure UTC datetime
    if "timestamp" in processed.columns:
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], utc=True)
        # Sort chronologically
        processed = processed.sort_values("timestamp").reset_index(drop=True)

    # Add text length statistics
    processed["char_length"] = processed["text"].str.len()
    processed["word_count"] = processed["text"].str.split().str.len()

    logger.info(f"Preprocessing complete. Final shape: {processed.shape}")
    logger.info(f"Posts with detected tickers: {processed['ticker'].notna().sum()} / {len(processed)}")

    return processed


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_strategy: str = "stratified",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.
    Options:
    - 'stratified': preserve class distribution (sentiment_label)
    - 'chronological': time-based split (prevents temporal leakage)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    set_seed(random_seed)

    if split_strategy == "chronological" and "timestamp" in df.columns:
        logger.info("Performing chronological split to prevent temporal leakage...")
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
    else:
        logger.info(f"Performing {split_strategy} split...")
        from sklearn.model_selection import train_test_split

        if "sentiment_label" in df.columns and df["sentiment_label"].notna().all():
            stratify = df["sentiment_label"]
        else:
            stratify = None

        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=stratify,
            random_state=random_seed,
        )

        val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
        temp_stratify = temp_df["sentiment_label"] if stratify is not None else None

        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_fraction_of_temp,
            stratify=temp_stratify,
            random_state=random_seed,
        )

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def run_preprocessing_pipeline(
    allow_synthetic: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end execution of data loading, preprocessing, and saving to disk.

    Args:
        allow_synthetic: If True, fall back to synthetic data when real data
            is unavailable. Defaults to False — research pipelines must use real data.
            Set to True ONLY for development and testing.
    """
    from src.data.ingestion import load_raw_data

    raw_df = load_raw_data(allow_synthetic=allow_synthetic)
    processed_df = preprocess_dataframe(raw_df)

    # Save full processed dataset
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    full_proc_path = DATA_PROC / "processed_reddit.parquet"
    processed_df.to_parquet(full_proc_path, index=False)
    logger.info(f"Saved processed dataset to {full_proc_path}")

    # Enforce strict separation from Reddit Gold evaluation pool
    annotation_file = Path("data/annotation_template.csv")
    if annotation_file.exists():
        ann_df = pd.read_csv(annotation_file)
        if "post_id" in ann_df.columns:
            excluded_ids = set(ann_df["post_id"].dropna().astype(str))
            n_before = len(processed_df)
            processed_df = processed_df[~processed_df["post_id"].astype(str).isin(excluded_ids)].copy()
            logger.info(
                f"Strict leakage control: excluded {n_before - len(processed_df)} gold evaluation "
                f"posts from distillation pool."
            )

    # Generate splits
    train_df, val_df, test_df = create_splits(processed_df, split_strategy="stratified")

    train_df.to_parquet(DATA_PROC / "train.parquet", index=False)
    val_df.to_parquet(DATA_PROC / "val.parquet", index=False)
    test_df.to_parquet(DATA_PROC / "test.parquet", index=False)

    logger.info("Saved train.parquet, val.parquet, test.parquet to data/processed/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = run_preprocessing_pipeline()
    print("\n--- Pipeline Summary ---")
    print(f"Train set: {len(train)} posts")
    print(f"Val set:   {len(val)} posts")
    print(f"Test set:  {len(test)} posts")
    if "sentiment_label" in train.columns:
        print("\nTrain class distribution:")
        print(train["sentiment_label"].value_counts(normalize=True).to_dict())
