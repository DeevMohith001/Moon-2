"""
Data ingestion: download and load Reddit financial sentiment datasets.

Dataset strategy:
  1. Try loading from local data/raw/ if already downloaded.
  2. Download from HuggingFace (public Reddit WSB/financial datasets).
  3. Standardise into a common schema.

IMPORTANT: Synthetic data is NEVER used for research results.
If real data is unavailable, this module raises DataUnavailableError
instead of silently generating synthetic data.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from src.utils.common import (
    DATA_RAW, DATA_INTERIM, DATA_PROC,
    load_config, get_logger, set_seed,
)

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────
class DataUnavailableError(RuntimeError):
    """Raised when no real dataset can be loaded or downloaded.
    Prevents silent fallback to synthetic data in research pipelines."""
    pass


# ── Standard Schema ───────────────────────────────────────────
RAW_SCHEMA_COLS = [
    "post_id", "timestamp", "subreddit", "title", "body",
    "score", "num_comments", "source",
]

PROCESSED_SCHEMA_COLS = [
    "post_id", "timestamp", "subreddit", "text", "ticker", "company",
    "sentiment_label", "sentiment_score", "model", "prompt_version",
    "confidence", "rationale",
]


# ── Download helpers ──────────────────────────────────────────


# ── Official Primary Dataset ──────────────────────────────────
PRIMARY_HF_DATASET = "emilpartow/reddit_finance_posts_sp500"

def load_reddit_hf_dataset(
    max_samples=None,
    use_streaming=True,
    split="train",
    seed=42,
):
    """
    Load the primary Reddit corpus from HuggingFace:
    emilpartow/reddit_finance_posts_sp500

    Dataset schema: id, title, text, created_utc, created_datetime, author,
    score, num_comments, upvote_ratio, flair, permalink, url, subreddit, company

    This is the OFFICIAL corpus for teacher labeling, student distillation,
    Reddit corpus analysis, and financial extension.

    Args:
        max_samples: If set, sample this many rows (reproducible via seed).
        use_streaming: If True (default), use streaming to prevent out-of-memory errors.
        split: Dataset split (default: 'train').
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with canonical schema. 'data_source' == 'REAL'.

    Raises:
        DataUnavailableError: If the dataset cannot be loaded.
    """
    import numpy as np
    logger.info(
        f"Loading primary HF dataset: {PRIMARY_HF_DATASET} "
        f"(split={split}, max_samples={max_samples}, streaming={use_streaming})"
    )
    try:
        from datasets import load_dataset
        if use_streaming or max_samples:
            ds = load_dataset(PRIMARY_HF_DATASET, split=split, streaming=True)
            records = []
            fetch_limit = int(max_samples * 2.5) if max_samples else 5000
            for row in ds:
                records.append(row)
                if len(records) >= fetch_limit:
                    break
            df = pd.DataFrame(records)
        else:
            try:
                ds = load_dataset(PRIMARY_HF_DATASET, split=split)
                df = ds.to_pandas()
            except Exception as e:
                logger.warning(f"Full dataset load failed ({e}), falling back to streaming...")
                ds = load_dataset(PRIMARY_HF_DATASET, split=split, streaming=True)
                records = []
                for row in ds:
                    records.append(row)
                    if len(records) >= 5000:
                        break
                df = pd.DataFrame(records)
        raw_count = len(df)
        logger.info(f"Loaded {raw_count} raw rows from {PRIMARY_HF_DATASET}")
        df = _normalize_sp500_dataset(df)
        if max_samples and max_samples < len(df):
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(df), size=max_samples, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
            logger.info(f"Sampled {len(df)} rows (seed={seed})")
        df["data_source"] = "REAL"
        df["source"] = PRIMARY_HF_DATASET
        logger.info(f"Final sp500 dataset: {len(df)} rows")
        return df
    except ImportError:
        raise DataUnavailableError(
            "The 'datasets' library is required: pip install datasets>=2.18.0"
        )
    except DataUnavailableError:
        raise
    except Exception as e:
        raise DataUnavailableError(
            f"Failed to load {PRIMARY_HF_DATASET}: {e}\n"
            "Ensure internet connectivity and the datasets library is installed."
        )


def _normalize_sp500_dataset(df):
    """
    Normalize emilpartow/reddit_finance_posts_sp500 to canonical schema.

    Mapping:
      id           -> post_id
      text         -> body   (sp500 uses 'text' for body/self-text)
      title        -> title
      created_utc  -> timestamp (Unix float seconds)
      subreddit    -> subreddit
      company      -> company
      score        -> score
      num_comments -> num_comments

    Returns: normalized DataFrame
    """
    result = df.copy()
    # 1. id -> post_id
    if "id" in result.columns:
        result["post_id"] = result["id"].astype(str)
    elif "post_id" not in result.columns:
        result["post_id"] = [f"sp500_{i}" for i in range(len(result))]
    # 2. text -> body
    if "text" in result.columns and "body" not in result.columns:
        result["body"] = result["text"].fillna("").astype(str)
    elif "body" not in result.columns:
        result["body"] = ""
    else:
        result["body"] = result["body"].fillna("").astype(str)
    # 3. title
    if "title" not in result.columns:
        result["title"] = ""
    else:
        result["title"] = result["title"].fillna("").astype(str)
    # 4. created_utc -> timestamp
    if "created_utc" in result.columns:
        ts_raw = result["created_utc"]
        if ts_raw.dtype in ("int64", "float64"):
            result["timestamp"] = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
        else:
            result["timestamp"] = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    elif "created_datetime" in result.columns:
        result["timestamp"] = pd.to_datetime(result["created_datetime"], utc=True, errors="coerce")
    else:
        result["timestamp"] = pd.NaT
    # 5. created_datetime
    if "created_datetime" not in result.columns:
        result["created_datetime"] = result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    # 6. numeric cols
    for col in ("score", "num_comments"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)
        else:
            result[col] = 0
    # 7. optional cols
    for col, default in [("upvote_ratio", 0.0), ("flair", None), ("permalink", None),
                          ("url", None), ("author", None), ("subreddit", "unknown"), ("company", None)]:
        if col not in result.columns:
            result[col] = default
    # 8. remove missing post_id
    n0 = len(result)
    result = result[result["post_id"].notna() & (result["post_id"].astype(str).str.strip() != "")].copy()
    n_missing = n0 - len(result)
    # 9. remove empty text
    combined = result["title"].str.strip() + " " + result["body"].str.strip()
    has_content = combined.str.strip().str.len() > 5
    n1 = len(result)
    result = result[has_content].copy()
    n_empty = n1 - len(result)
    # 10. dedup by post_id
    n2 = len(result)
    result = result.drop_duplicates(subset=["post_id"]).copy()
    n_dedup = n2 - len(result)
    # 11. sort chronologically
    if result["timestamp"].notna().any():
        result = result.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"sp500 normalization: {n0} -> {len(result)} rows "
        f"(missing_id={n_missing}, empty_text={n_empty}, dedup={n_dedup})"
    )
    return result


def generate_sp500_dataset_stats(df):
    """
    Generate comprehensive statistics for the sp500 Reddit dataset.
    Records loaded, removed, deduplicated, and retained row counts.

    Returns:
        Dictionary with dataset statistics for reproducibility documentation.
    """
    stats = {
        "total_rows": len(df),
        "data_source": str(df["data_source"].unique().tolist() if "data_source" in df.columns else ["UNKNOWN"]),
        "columns": list(df.columns),
        "dataset_url": "https://huggingface.co/datasets/emilpartow/reddit_finance_posts_sp500",
    }
    if "subreddit" in df.columns:
        stats["subreddit_distribution"] = df["subreddit"].value_counts().to_dict()
        stats["unique_subreddits"] = int(df["subreddit"].nunique())
    if "company" in df.columns:
        stats["unique_companies"] = int(df["company"].nunique())
        stats["top_20_companies"] = df["company"].value_counts().head(20).to_dict()
    if "timestamp" in df.columns:
        ts = df["timestamp"].dropna()
        if len(ts) > 0:
            stats["date_range_start"] = str(ts.min())
            stats["date_range_end"] = str(ts.max())
            stats["date_range_days"] = int((ts.max() - ts.min()).days)
    if "score" in df.columns:
        stats["score_mean"] = float(df["score"].mean())
        stats["score_median"] = float(df["score"].median())
        stats["score_max"] = int(df["score"].max())
    if "num_comments" in df.columns:
        stats["num_comments_mean"] = float(df["num_comments"].mean())
    if "title" in df.columns and "body" in df.columns:
        combined = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
        stats["text_length_mean_chars"] = float(combined.str.len().mean())
        stats["text_length_median_chars"] = float(combined.str.len().median())
        stats["text_length_p95_chars"] = float(combined.str.len().quantile(0.95))
        stats["text_length_max_chars"] = int(combined.str.len().max())
    return stats

def download_reddit_dataset() -> pd.DataFrame:
    """
    Download a public Reddit financial dataset from HuggingFace.

    IMPORTANT: This function does NOT fall back to synthetic data.
    If no real dataset can be downloaded, it raises DataUnavailableError.
    """
    logger.info("Attempting to download Reddit financial dataset...")

    # Delegate to primary dataset loader
    return load_reddit_hf_dataset()


def _standardize_hf_reddit(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a HuggingFace Reddit dataset to our standard schema."""
    logger.info(f"Standardizing dataset from {source}. Columns: {list(df.columns)}")

    result = pd.DataFrame()

    # Map common column names
    col_map = {
        "id": "post_id", "name": "post_id",
        "created_utc": "timestamp", "created": "timestamp",
        "timestamp": "timestamp",
        "subreddit": "subreddit",
        "title": "title",
        "selftext": "body", "body": "body", "text": "body",
        "score": "score", "ups": "score",
        "num_comments": "num_comments",
    }

    for src_col, dst_col in col_map.items():
        if src_col in df.columns and dst_col not in result.columns:
            result[dst_col] = df[src_col]

    # Ensure post_id exists
    if "post_id" not in result.columns:
        result["post_id"] = [f"post_{i}" for i in range(len(result))]

    # Parse timestamp
    if "timestamp" in result.columns:
        ts = result["timestamp"]
        if ts.dtype in ("int64", "float64"):
            result["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        else:
            result["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    # Defaults
    if "subreddit" not in result.columns:
        result["subreddit"] = "wallstreetbets"
    if "title" not in result.columns:
        result["title"] = ""
    if "body" not in result.columns:
        result["body"] = ""
    if "score" not in result.columns:
        result["score"] = 0
    if "num_comments" not in result.columns:
        result["num_comments"] = 0

    result["source"] = source
    result["data_source"] = "REAL"
    result = result[[c for c in RAW_SCHEMA_COLS + ["data_source"] if c in result.columns]]

    logger.info(f"Standardized to {len(result)} rows, columns: {list(result.columns)}")
    return result


def generate_development_dataset() -> pd.DataFrame:
    """
    Generate a small realistic development dataset for testing the pipeline.

    ⚠️  IMPORTANT: These are illustrative Reddit-style posts — NOT real data.
    This function is ONLY for:
      - Unit tests
      - Smoke tests
      - Development pipeline testing
    Results from this data must NEVER be reported as research results.

    The returned DataFrame includes a 'data_source' column set to 'SYNTHETIC'
    so downstream code can verify data provenance.
    """
    set_seed(42)
    rng = np.random.default_rng(42)

    logger.warning(
        "⚠️  GENERATING SYNTHETIC DEVELOPMENT DATASET. "
        "This data must NOT be used for research results."
    )

    # Realistic examples spanning sentiment types
    examples = [
        # Bullish
        {"title": "NVDA to the moon 🚀🚀🚀", "body": "Nvidia earnings are going to crush it. AI demand is insane. Loading up on calls.", "ticker": "NVDA", "sentiment": "BULLISH"},
        {"title": "AAPL looking strong", "body": "Apple's new product lineup is incredible. iPhone sales going to be huge this quarter. Buying more shares.", "ticker": "AAPL", "sentiment": "BULLISH"},
        {"title": "TSLA bull case", "body": "Tesla FSD is getting so much better. Robotaxi revenue is going to be massive. This stock is going to $500.", "ticker": "TSLA", "sentiment": "BULLISH"},
        {"title": "GME diamond hands 💎🙌", "body": "Not selling. The squeeze hasn't squozen. HODL to the moon.", "ticker": "GME", "sentiment": "BULLISH"},
        {"title": "MSFT cloud growth incredible", "body": "Azure revenue up 30%. GitHub Copilot adoption exploding. This is a $400 stock easy.", "ticker": "MSFT", "sentiment": "BULLISH"},
        {"title": "AMD about to pop", "body": "AMD gaining server market share like crazy. Their new chips crush Intel. Going long.", "ticker": "AMD", "sentiment": "BULLISH"},
        {"title": "AMZN earnings play", "body": "AWS margins improving, advertising growing 25%+. This is undervalued at current levels.", "ticker": "AMZN", "sentiment": "BULLISH"},
        {"title": "META VR play", "body": "Everyone sleeping on Meta's AI. Reels monetization improving. Revenue reacceleration incoming.", "ticker": "META", "sentiment": "BULLISH"},
        {"title": "GOOGL AI moat", "body": "Google has the best AI infrastructure. Gemini is incredible. Search isn't going anywhere. BUY.", "ticker": "GOOGL", "sentiment": "BULLISH"},
        {"title": "SPY calls printing", "body": "Market going higher. Fed done hiking. Soft landing confirmed. Buy every dip.", "ticker": "SPY", "sentiment": "BULLISH"},

        # Bearish
        {"title": "TSLA is overvalued garbage", "body": "PE ratio is insane. Margins declining. Competition catching up. This is a $100 stock.", "ticker": "TSLA", "sentiment": "BEARISH"},
        {"title": "AAPL peaked", "body": "iPhone growth is done. China sales declining. Services can't carry this valuation forever. Selling.", "ticker": "AAPL", "sentiment": "BEARISH"},
        {"title": "Market crash incoming", "body": "Yield curve inverted for 2 years. Recession is coming. Get out of stocks now. Buy puts on SPY.", "ticker": "SPY", "sentiment": "BEARISH"},
        {"title": "NVDA bubble", "body": "AI hype is just like the dot-com bubble. When it pops, NVDA goes back to $200. Shorting this.", "ticker": "NVDA", "sentiment": "BEARISH"},
        {"title": "GME bagholders unite", "body": "This company has no real plan. Revenue declining. The squeeze is over. Cut your losses.", "ticker": "GME", "sentiment": "BEARISH"},
        {"title": "COIN going to zero", "body": "Crypto winter isn't over. SEC crackdown coming. Revenue down 50%. Avoid this trash.", "ticker": "COIN", "sentiment": "BEARISH"},
        {"title": "BA quality disasters", "body": "Boeing can't make a safe plane. Orders getting cancelled. Debt is massive. Short this.", "ticker": "BA", "sentiment": "BEARISH"},
        {"title": "NFLX subscriber peak", "body": "Netflix is losing content. Password sharing crackdown won't help. Streaming wars over. Sell.", "ticker": "NFLX", "sentiment": "BEARISH"},
        {"title": "DIS losing money on streaming", "body": "Disney+ burning cash. Parks growth slowing. Bob Iger can't save this. Bearish.", "ticker": "DIS", "sentiment": "BEARISH"},
        {"title": "INTC dead company walking", "body": "Intel lost the foundry race. AMD and ARM eating their lunch. Dividend at risk. Puts.", "ticker": "INTC", "sentiment": "BEARISH"},

        # Neutral
        {"title": "AAPL earnings tomorrow", "body": "Anyone playing earnings? Could go either way. Waiting for the numbers before deciding.", "ticker": "AAPL", "sentiment": "NEUTRAL"},
        {"title": "What do you think about MSFT?", "body": "Looking at the chart. Seems fairly valued at this level. Not sure if I should buy or wait.", "ticker": "MSFT", "sentiment": "NEUTRAL"},
        {"title": "Market open thread", "body": "What's everyone watching today? I'm keeping an eye on tech earnings this week.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "AMZN stock split discussion", "body": "Does the stock split change anything fundamentally? Just more shares at lower price right?", "ticker": "AMZN", "sentiment": "NEUTRAL"},
        {"title": "Beginner question about options", "body": "Can someone explain how calls work? I see people talking about TSLA options. How risky is it?", "ticker": "TSLA", "sentiment": "NEUTRAL"},
        {"title": "DD: NVDA competitive landscape", "body": "Looking at the GPU market objectively. NVDA dominates but AMD is gaining. Here's the analysis...", "ticker": "NVDA", "sentiment": "NEUTRAL"},
        {"title": "Portfolio rebalancing", "body": "Moving from 80/20 stocks/bonds to 70/30. Not making a market call just reducing risk as I get older.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "META vs GOOGL comparison", "body": "Both seem fairly priced. META has better margins but GOOGL more diversified. Tough call.", "ticker": "META", "sentiment": "NEUTRAL"},
        {"title": "Fed meeting notes", "body": "Powell said data-dependent. Dot plot unchanged. Market didn't react much. Staying the course.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "Tax loss harvesting strategy", "body": "Selling some losers to offset gains. Not a reflection on the companies just tax strategy.", "ticker": "SPY", "sentiment": "NEUTRAL"},

        # Sarcastic / challenging
        {"title": "I'm financially ruined", "body": "YOLO'd my entire savings into GME weekly calls. Down 95%. Best investment ever 🤡", "ticker": "GME", "sentiment": "BEARISH"},
        {"title": "DD: Trust me bro", "body": "My cousin works at Apple and he said next iPhone will have AI. Buy calls. Source: trust me bro.", "ticker": "AAPL", "sentiment": "BULLISH"},
        {"title": "Loss porn: -$50k on TSLA puts", "body": "Elon tweeted and my puts went to zero. I'm never betting against this man again. Inverse me.", "ticker": "TSLA", "sentiment": "BEARISH"},
        {"title": "GUH moment", "body": "Accidentally bought 10x more NVDA calls than intended. Up 200% though so we good 😂🚀", "ticker": "NVDA", "sentiment": "BULLISH"},
        {"title": "Inverse Cramer working again", "body": "Cramer said buy COIN so I shorted it. Already up 15%. This strategy never fails.", "ticker": "COIN", "sentiment": "BEARISH"},
    ]

    # Templates for diverse text generation
    bullish_templates = [
        ("{ticker} to the moon! Just loaded up on {strike} calls expiring next month. Strong fundamentals and momentum.", "Earnings beat expected. Demand is through the roof. PT ${target}."),
        ("Why I'm extremely bullish on {ticker}: {reason}", "Position: {shares} shares and {calls} call contracts. Can't see this going anywhere but up."),
        ("{ticker} breaking out of resistance right now! 🚀🚀", "Huge volume spike at market open. Institutional buyers stepping in."),
        ("Deep dive DD into {ticker}: Massive moat and accelerating revenue", "Operating margins expanded by {pct}%. Cloud/hardware backlog is solid."),
        ("Never bet against {ticker}. Diamond hands 💎🙌", "Holding through the volatility. The long-term thesis has never looked better."),
        ("Analyst upgrade on {ticker} to Strong Buy with ${target} target", "Undervalued compared to sector peers. Free cash flow yield is attractive."),
        ("Bought the dip on {ticker} today at ${price}", "Great entry point. Rebound looks imminent as RSI shows oversold bounce."),
        ("Massive insider buying reported for {ticker}", "Executive team just purchased ${amount}k worth of stock on open market."),
    ]

    bearish_templates = [
        ("{ticker} is completely overvalued and ready to crash", "P/E multiple is disconnected from reality. Margins deteriorating rapidly. Shorting."),
        ("Why you should sell {ticker} before next earnings", "Supply chain disruptions and lowering guidance. Competition eating market share."),
        ("Loss porn: down {pct}% on {ticker} weekly calls 🤡", "Theta gang destroyed my portfolio. Should have bought puts instead."),
        ("{ticker} broke critical support level today", "High volume selloff. Next major support isn't until ${target}. Downside target in play."),
        ("Terrible guidance from {ticker} management", "Revenue growth slowed to {pct}%. Debt load is too high in this interest rate environment."),
        ("Puts on {ticker} printing money today 🩸📉", "Broke key 200 EMA. Bear flag breakdown confirmed. Added to my short position."),
        ("Insiders dumping shares of {ticker}", "CEO sold ${amount}M of stock this week. They know what's coming next quarter."),
        ("Macro headwinds will crush {ticker}'s forward earnings", "Consumer spending tightening, valuation is untenable at {multiple}x sales."),
    ]

    neutral_templates = [
        ("{ticker} upcoming earnings discussion thread", "What are your expectations for Q{quarter}? Implied volatility is pricing a {pct}% move."),
        ("Looking at {ticker} chart - consolidation pattern forming", "Trading in a tight range between ${low} and ${high}. Waiting for breakout confirmation."),
        ("Portfolio rebalancing: Thinking about trimmed position in {ticker}", "Not bearish, but looking to take some profits and reallocate to cash or index."),
        ("What is your fair value estimate for {ticker}?", "DCF model suggests around ${target}, but discount rate sensitivity is high. Opinions?"),
        ("Options flow unusual activity in {ticker}", "Large block trade seen in {strike} strikes. Could be a hedge or straddle position."),
        ("Comparing {ticker} vs sector competitors for long term hold", "Both have pros and cons. Valuation seems fair at current prices."),
        ("Fed interest rate decision impact on {ticker}", "Neutral reaction so far. Watching how Treasury yields settle before adding."),
        ("Covered call strategy on {ticker}", "Selling ${strike} calls against my position to generate yield during this sideways trend."),
    ]

    tickers = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "GME", "SPY", "COIN", "PLTR", "INTC", "BA"]
    reasons = [
        "AI infrastructure demand is unmatched", "services revenue recurring cash cows",
        "market expansion and autonomous rollouts", "enterprise enterprise software lock-in",
        "logistics network efficiencies", "monetization of digital ad impressions",
        "datacenter market share capture", "custom silicon roadmap execution"
    ]

    all_posts = []
    base_date = datetime(2023, 1, 1)
    post_idx = 0

    categories = [
        ("BULLISH", bullish_templates),
        ("BEARISH", bearish_templates),
        ("NEUTRAL", neutral_templates),
    ]

    for sentiment_label, templates in categories:
        for tpl_title, tpl_body in templates:
            for ticker in tickers:
                # 3-5 unique posts per combination
                for _ in range(rng.integers(3, 6)):
                    post_idx += 1
                    days_offset = int(rng.integers(0, 700))
                    hour = int(rng.integers(0, 24))
                    minute = int(rng.integers(0, 60))
                    ts = base_date + pd.Timedelta(days=days_offset, hours=hour, minutes=minute)

                    # Dynamic format values
                    params = {
                        "ticker": f"${ticker}" if rng.random() > 0.4 else ticker,
                        "strike": int(rng.integers(50, 600)),
                        "target": int(rng.integers(50, 800)),
                        "price": int(rng.integers(30, 500)),
                        "low": int(rng.integers(80, 200)),
                        "high": int(rng.integers(210, 450)),
                        "pct": int(rng.integers(5, 75)),
                        "shares": int(rng.integers(50, 2000)),
                        "calls": int(rng.integers(2, 50)),
                        "amount": int(rng.integers(10, 500)),
                        "multiple": int(rng.integers(15, 60)),
                        "quarter": int(rng.integers(1, 5)),
                        "reason": rng.choice(reasons),
                    }

                    title = tpl_title.format(**params)
                    body = tpl_body.format(**params)

                    # Subreddit styling
                    sub = rng.choice(["wallstreetbets", "stocks", "investing", "StockMarket"])

                    all_posts.append({
                        "post_id": f"post_{post_idx:05d}",
                        "timestamp": ts,
                        "subreddit": sub,
                        "title": title,
                        "body": body,
                        "score": int(rng.integers(5, 8500)),
                        "num_comments": int(rng.integers(1, 650)),
                        "source": "SYNTHETIC_DEV_v1",
                        "data_source": "SYNTHETIC",
                        "_ticker": ticker,
                        "_sentiment": sentiment_label,
                    })

    df = pd.DataFrame(all_posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Generated {len(df)} diverse SYNTHETIC development posts")
    return df


# ── Main ingestion logic ──────────────────────────────────────
def load_raw_data(
    force_download: bool = False,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Load raw Reddit data. Uses cached version if available.

    Args:
        force_download: If True, re-download even if cached file exists.
        allow_synthetic: If True, fall back to synthetic data when real data
            is unavailable. This should ONLY be used for development and testing,
            NEVER for research results.

    Returns:
        DataFrame with raw Reddit posts.

    Raises:
        DataUnavailableError: If no real data can be loaded and allow_synthetic=False.
    """
    raw_file = DATA_RAW / "reddit_posts.parquet"

    if raw_file.exists() and not force_download:
        logger.info(f"Loading cached raw data from {raw_file}")
        df = pd.read_parquet(raw_file)
        logger.info(f"Loaded {len(df)} posts from cache")
        # Tag data source if not already tagged
        if "data_source" not in df.columns:
            source_val = df.get("source", pd.Series(["unknown"])).iloc[0]
            if "synthetic" in str(source_val).lower():
                df["data_source"] = "SYNTHETIC"
            else:
                df["data_source"] = "REAL"

        # If cached data is synthetic but synthetic data is not permitted, do not return it
        if not allow_synthetic and (df["data_source"] == "SYNTHETIC").any():
            logger.warning(
                "Cached raw data contains SYNTHETIC posts, but allow_synthetic=False. "
                "Attempting to download real dataset instead..."
            )
        else:
            return df

    # Try downloading real data
    try:
        df = download_reddit_dataset()
        df["data_source"] = "REAL"
    except DataUnavailableError:
        if allow_synthetic:
            logger.warning(
                "⚠️  Real data unavailable. Using SYNTHETIC development dataset. "
                "Results from this data must NOT be reported as research results."
            )
            df = generate_development_dataset()
        else:
            raise

    # Save raw
    df.to_parquet(raw_file, index=False)
    logger.info(f"Saved {len(df)} posts to {raw_file}")

    return df


def assert_real_data(df: pd.DataFrame) -> None:
    """
    Guard function: raises DataUnavailableError if DataFrame contains
    synthetic data. Use this at the start of any research pipeline.
    """
    if "data_source" in df.columns:
        if (df["data_source"] == "SYNTHETIC").any():
            raise DataUnavailableError(
                "This DataFrame contains SYNTHETIC data. "
                "Research results cannot be generated from synthetic data. "
                "Please provide a real Reddit dataset."
            )
    if "source" in df.columns:
        sources = df["source"].unique()
        if any("synthetic" in str(s).lower() for s in sources):
            raise DataUnavailableError(
                f"This DataFrame was generated from synthetic source(s): {sources}. "
                "Research results cannot be generated from synthetic data."
            )


def load_fiqa_dataset(subtask: Optional[str] = None) -> dict:
    """
    Load the FiQA sentiment dataset for cross-dataset evaluation.

    Paper Protocol Alignment (Deng et al. 2023 Table 1 & Table 2 / FiQA-2018):
    - Two subtasks: 'news' (format == 'headline') and 'post' (format == 'post')
    - Uses continuous sentiment score (column 'sentiment_score' or 'score')
    - Removes examples with score == 0 (ambiguous / neutral examples)
    - Removes multi-stock examples (texts mentioning multiple stocks/entities)
    - Evaluates positive (score > 0) vs negative (score < 0) only
    - Evaluates on 80/10/10 train/val/test splits

    Args:
        subtask: Optional subtask filter: 'news' (headlines) or 'post' (microblogs).
                 If None, returns full dataset splits.

    Returns:
        dict with 'train', 'validation', 'test' DataFrames.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pauri32/fiqa-2018")
        splits = {}
        for split_name in ds:
            df = ds[split_name].to_pandas()
            # Standardize text column
            if "sentence" in df.columns:
                df = df.rename(columns={"sentence": "text"})

            # Optional subtask filtering
            if subtask is not None and "format" in df.columns:
                sub_norm = subtask.strip().lower()
                if sub_norm in ("news", "headline", "headlines"):
                    df = df[df["format"] == "headline"].copy()
                elif sub_norm in ("post", "posts", "microblog"):
                    df = df[df["format"] == "post"].copy()

            score_col = None
            if "score" in df.columns:
                score_col = "score"
            elif "sentiment_score" in df.columns:
                score_col = "sentiment_score"

            if score_col is not None:
                initial_count = len(df)

                # Paper protocol: remove score == 0 (ambiguous examples)
                df = df[df[score_col] != 0].copy()
                logger.info(
                    f"FiQA {split_name} (subtask={subtask}): removed {initial_count - len(df)} "
                    f"examples with score == 0"
                )

                # Paper protocol: remove multi-stock examples
                if "target" in df.columns:
                    multi_stock_mask = df["target"].astype(str).str.contains(r"[,;/]")
                    multi_count = multi_stock_mask.sum()
                    df = df[~multi_stock_mask].copy()
                    logger.info(
                        f"FiQA {split_name} (subtask={subtask}): removed {multi_count} multi-stock examples"
                    )

                # Paper protocol: map continuous score to binary labels
                df["sentiment_label"] = df[score_col].apply(
                    lambda x: "BULLISH" if x > 0 else "BEARISH"
                )
                df["continuous_score"] = df[score_col]

            df["data_source"] = "REAL"
            splits[split_name] = df.reset_index(drop=True)
            logger.info(f"FiQA {split_name} (subtask={subtask}): {len(df)} samples after filtering")
        return splits
    except Exception as e:
        logger.warning(f"Could not load FiQA dataset: {e}")
        return {}


if __name__ == "__main__":
    set_seed(42)
    # For development, allow synthetic fallback
    df = load_raw_data(force_download=True, allow_synthetic=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data source: {df['data_source'].unique()}")
    print(f"\nSample:\n{df.head()}")
    if "timestamp" in df.columns:
        print(f"\nDate range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    if "subreddit" in df.columns:
        print(f"\nSubreddits: {df['subreddit'].value_counts().to_dict()}")
