"""End-to-end runner for the modernized Deng et al. paper pipeline.

This is a reproduction-oriented orchestration script. It preserves the paper's
core flow, while using this repository's modern provider abstraction and
DistilBERT-compatible student model instead of PaLM-540B and Charformer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.preprocessing import run_preprocessing_pipeline
from src.distillation.dataset import filter_by_consistency
from src.distillation.evaluate import evaluate_regression_student
from src.distillation.train import train_regression_student
from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels
from src.llm.client import get_llm_provider, load_dotenv_if_available
from src.llm.labeling import generate_teacher_weak_labels_dataset
from src.llm.prompts import PromptManager
from src.utils.common import DATA_PROC, RESULTS_DIR
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    train_path: Path = DATA_PROC / "train.parquet"
    val_path: Path = DATA_PROC / "val.parquet"
    test_path: Path = DATA_PROC / "test.parquet"
    provider: Optional[str] = None
    model_name: Optional[str] = None
    num_paths: int = 8
    consistency_threshold: int = 5
    max_train_samples: Optional[int] = 100
    max_val_samples: Optional[int] = 30
    epochs: int = 3
    batch_size: int = 16
    backbone_name: str = "distilbert-base-uncased"
    smoke: bool = False
    allow_synthetic_data: bool = False


def _ensure_processed_splits(config: PipelineConfig) -> None:
    required = [config.train_path, config.val_path, config.test_path]
    if all(path.exists() for path in required):
        return

    missing = ", ".join(str(path) for path in required if not path.exists())
    if not config.smoke and not config.allow_synthetic_data:
        raise FileNotFoundError(
            "Required processed dataset split(s) are missing: "
            f"{missing}. Run `python -m src.data.preprocessing` with a real dataset first, "
            "or use `--smoke` for an offline synthetic development check."
        )

    logger.warning(
        "Processed splits are missing. Creating clearly marked synthetic development "
        "splits because --smoke or --allow-synthetic-data was requested."
    )
    run_preprocessing_pipeline(allow_synthetic=True)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _with_sample_limit(df: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    return df.iloc[:limit].copy() if limit else df.copy()


def run_paper_pipeline(config: PipelineConfig) -> dict:
    """Run teacher weak labeling, aggregation, filtering, distillation, and evaluation."""
    load_dotenv_if_available()
    set_seed(42)
    _ensure_processed_splits(config)

    train_df = _with_sample_limit(_read_frame(config.train_path), config.max_train_samples)
    val_df = _with_sample_limit(_read_frame(config.val_path), config.max_val_samples)
    test_df = _read_frame(config.test_path)

    provider = get_llm_provider(provider=config.provider, model_name=config.model_name)
    prompt_manager = PromptManager("sentiment_prompt")

    logger.info("Generating train teacher reasoning paths...")
    train_paths = generate_teacher_weak_labels_dataset(
        df=train_df,
        provider=provider,
        prompt_manager=prompt_manager,
        num_paths=config.num_paths,
        max_samples=config.max_train_samples,
        include_cot=True,
    )
    train_weak = aggregate_paths_dataframe(train_paths, original_df=train_df)
    save_weak_labels(train_weak, filename="paper_pipeline_train_weak_labels.csv")

    logger.info("Generating validation teacher reasoning paths...")
    val_paths = generate_teacher_weak_labels_dataset(
        df=val_df,
        provider=provider,
        prompt_manager=prompt_manager,
        num_paths=config.num_paths,
        max_samples=config.max_val_samples,
        include_cot=True,
    )
    val_weak = aggregate_paths_dataframe(val_paths, original_df=val_df)
    save_weak_labels(val_weak, filename="paper_pipeline_val_weak_labels.csv")

    train_filtered, train_filter_stats = filter_by_consistency(
        train_weak,
        threshold=config.consistency_threshold,
        total_paths=config.num_paths,
    )
    val_filtered, val_filter_stats = filter_by_consistency(
        val_weak,
        threshold=config.consistency_threshold,
        total_paths=config.num_paths,
    )

    if train_filtered.empty:
        raise RuntimeError("Consistency filtering removed all training examples.")
    if val_filtered.empty:
        logger.warning("Validation filtering removed all rows; using unfiltered validation labels.")
        val_filtered = val_weak

    model, tokenizer, history = train_regression_student(
        train_df=train_filtered,
        val_df=val_filtered,
        backbone_name=config.backbone_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        save_name="distilbert_student_regression",
    )

    metrics = evaluate_regression_student(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        val_df=val_filtered,
        batch_size=config.batch_size,
        save_prefix="paper_pipeline_student",
    )

    summary = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "train_filter_stats": train_filter_stats,
        "val_filter_stats": val_filter_stats,
        "train_history": history,
        "test_metrics": metrics,
    }

    out_path = RESULTS_DIR / "metrics" / "paper_pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved paper pipeline summary to {out_path}")
    return summary


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the modernized paper reproduction pipeline.")
    parser.add_argument("--provider", default=None, help="LLM provider: mock, openai, gemini, anthropic, or local.")
    parser.add_argument("--model-name", default=None, help="Provider model identifier.")
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--consistency-threshold", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=100)
    parser.add_argument("--max-val-samples", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-name", default="distilbert-base-uncased")
    parser.add_argument("--train-path", type=Path, default=DATA_PROC / "train.parquet")
    parser.add_argument("--val-path", type=Path, default=DATA_PROC / "val.parquet")
    parser.add_argument("--test-path", type=Path, default=DATA_PROC / "test.parquet")
    parser.add_argument("--smoke", action="store_true", help="Use tiny limits for a quick local check.")
    parser.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="Development only: create/use synthetic splits if real processed data is unavailable.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.provider = args.provider or "mock"
        args.max_train_samples = min(args.max_train_samples, 10)
        args.max_val_samples = min(args.max_val_samples, 5)
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 4)

    return PipelineConfig(**vars(args))


if __name__ == "__main__":
    result = run_paper_pipeline(parse_args())
    print(json.dumps(result["test_metrics"], indent=2, default=str))
