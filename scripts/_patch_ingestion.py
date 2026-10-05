"""
Patch script: insert load_reddit_hf_dataset() and related functions into ingestion.py
"""
import pathlib

new_func = '''

# ── Official Primary Dataset ──────────────────────────────────
PRIMARY_HF_DATASET = "emilpartow/reddit_finance_posts_sp500"

def load_reddit_hf_dataset(
    max_samples=None,
    use_streaming=False,
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
        use_streaming: If True, use streaming for low-memory environments.
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
        f"(split={split}, max_samples={max_samples})"
    )
    try:
        from datasets import load_dataset
        if use_streaming and max_samples:
            ds = load_dataset(PRIMARY_HF_DATASET, split=split, streaming=True)
            records = []
            fetch_limit = int(max_samples * 2.5)
            for row in ds:
                records.append(row)
                if len(records) >= fetch_limit:
                    break
            df = pd.DataFrame(records)
        else:
            ds = load_dataset(PRIMARY_HF_DATASET, split=split)
            df = ds.to_pandas()
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
            f"Failed to load {PRIMARY_HF_DATASET}: {e}\\n"
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

'''

ingestion_path = pathlib.Path("src/data/ingestion.py")
content = ingestion_path.read_text(encoding="utf-8")

insert_marker = "def download_reddit_dataset() ->"
if insert_marker not in content:
    print("ERROR: Marker not found in ingestion.py")
    exit(1)

idx = content.index(insert_marker)
new_content = content[:idx] + new_func + content[idx:]

# Update download_reddit_dataset to call load_reddit_hf_dataset as primary source
old_download_body = '''    errors = []

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset

        # Try multiple dataset sources in order of preference
        hf_datasets = [
            ("emilpartow/reddit_finance_posts_apple-tesla-microsoft", None),
            ("SocialGrep/reddit-r-wallstreetbets-posts-daily", None),
        ]

        for ds_name, ds_config in hf_datasets:
            try:
                logger.info(f"Trying HuggingFace dataset: {ds_name}")
                kwargs = {}
                if ds_config:
                    ds = load_dataset(ds_name, ds_config, **kwargs)
                else:
                    ds = load_dataset(ds_name, **kwargs)
                # Get the first available split
                split_name = list(ds.keys())[0]
                df = ds[split_name].to_pandas()
                logger.info(f"Loaded {len(df)} rows from {ds_name} ({split_name})")
                return _standardize_hf_reddit(df, ds_name)
            except Exception as e:
                logger.warning(f"Could not load {ds_name}: {e}")
                errors.append(f"{ds_name}: {e}")
                continue

    except ImportError:
        errors.append("'datasets' library not installed (pip install datasets)")
        logger.warning("datasets library not installed")

    # NO SYNTHETIC FALLBACK — raise error with actionable guidance
    raise DataUnavailableError(
        "No real Reddit dataset could be loaded.\\n"
        "Attempted sources and errors:\\n"
        + "\\n".join(f"  - {e}" for e in errors) + "\\n\\n"
        "To proceed, you must do ONE of the following:\\n"
        "  1. Install the 'datasets' library: pip install datasets\\n"
        "  2. Place a real Reddit dataset at: data/raw/reddit_posts.parquet\\n"
        "  3. For development/testing ONLY, use generate_development_dataset() explicitly\\n"
        "     (results from synthetic data must NEVER be reported as research results)"
    )'''

new_download_body = '''    # Delegate to primary dataset loader
    return load_reddit_hf_dataset()'''

if old_download_body in new_content:
    new_content = new_content.replace(old_download_body, new_download_body)
    print("Updated download_reddit_dataset body")
else:
    print("WARNING: Could not update download_reddit_dataset body (already updated or different)")

ingestion_path.write_text(new_content, encoding="utf-8")
print(f"Done: ingestion.py updated, new size: {len(new_content)} bytes")

# Quick verification
import importlib.util
spec = importlib.util.spec_from_file_location("ingestion", "src/data/ingestion.py")
mod = importlib.util.module_from_spec(spec)
try:
    # Just check if the function exists syntactically
    import ast
    tree = ast.parse(new_content)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    print(f"Functions defined: {[f for f in func_names if 'reddit' in f.lower() or 'sp500' in f.lower() or 'dataset' in f.lower()]}")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
