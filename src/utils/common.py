"""
Shared utilities: config loading, logging, paths, reproducibility, and caching.
"""

import os
import sys
import json
import random
import logging
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.config import load_config, load_prompt_config

# ── Project directories ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_EXT     = PROJECT_ROOT / "data" / "external"
DATA_CACHE   = PROJECT_ROOT / "data" / "cache"

MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
REPORTS_DIR  = PROJECT_ROOT / "reports"
CONFIGS_DIR  = PROJECT_ROOT / "configs"
PROMPTS_DIR  = PROJECT_ROOT / "prompts"

# Ensure all primary working directories exist
for d in [
    DATA_RAW, DATA_INTERIM, DATA_PROC, DATA_EXT, DATA_CACHE,
    MODELS_DIR,
    RESULTS_DIR / "metrics", RESULTS_DIR / "figures",
    RESULTS_DIR / "predictions", RESULTS_DIR / "error_analysis",
    OUTPUTS_DIR / "models", OUTPUTS_DIR / "metrics",
    OUTPUTS_DIR / "figures", OUTPUTS_DIR / "reports",
    REPORTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def load_all_configs() -> dict:
    """Load main config merged with data, model, training, and llm configs."""
    cfg = {}
    for cfg_name in ["config", "data", "model", "training", "experiments", "llm_config"]:
        try:
            sub = load_config(cfg_name)
            cfg.update(sub)
        except FileNotFoundError:
            pass
    return cfg


# ── Caching ───────────────────────────────────────────────────
def cache_key(*args) -> str:
    """Create a deterministic cache key from arguments."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_cache(cache_dir: Path, key: str):
    """Load cached JSON result if it exists."""
    path = cache_dir / f"{key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache_dir: Path, key: str, data):
    """Save data to JSON cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


# ── Experiment tracking ───────────────────────────────────────
def log_experiment(experiment_id: str, metrics: dict, config: dict = None):
    """Append experiment result to the tracking CSV in both results/ and outputs/metrics/."""
    import pandas as pd

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    if config:
        row["config"] = json.dumps(config, default=str)

    df_new = pd.DataFrame([row])
    for target_dir in [RESULTS_DIR, OUTPUTS_DIR / "metrics"]:
        target_file = target_dir / "experiment_results.csv"
        if target_file.exists():
            try:
                df_old = pd.read_csv(target_file)
                df = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df = df_new
        else:
            df = df_new
        df.to_csv(target_file, index=False)

    return row
