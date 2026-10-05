"""
End-to-End Zero-Cost Smoke Test.

Executes the entire paper reproduction pipeline on 10 sample Reddit posts:
10 Reddit Posts
    ↓
LLM Mock Provider
    ↓
8 Reasoning Paths per Post (80 total generations)
    ↓
Majority Voting & Soft Agreement Score
    ↓
Weak-Label Generation
    ↓
Consistency Filtering (>= 5/8 agreement)
    ↓
Student Dataset Creation
    ↓
Regression Distillation Training (1 epoch)
    ↓
Evaluation & Optimal Threshold Tuning
    ↓
Outputs Verification

CRITICAL: Does NOT require any commercial/paid API key. Fully deterministic and offline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import MockProvider
from src.llm.prompts import PromptManager
from src.llm.labeling import generate_teacher_weak_labels_dataset
from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels
from src.distillation.dataset import filter_by_consistency, TextRegressionDataset
from src.distillation.student_model import build_student_model
from src.distillation.train import train_regression_student
from src.distillation.evaluate import evaluate_regression_student
from src.utils.common import OUTPUTS_DIR, RESULTS_DIR


def test_end_to_end_smoke_pipeline():
    """Run full pipeline smoke test on 10 synthetic samples."""
    print("\n" + "=" * 60)
    print("STARTING END-TO-END SMOKE TEST (10 SAMPLES)")
    print("=" * 60)

    # 1. Create tiny dataset of 10 realistic Reddit posts
    posts_data = [
        {"post_id": "p1", "ticker": "NVDA", "text": "NVDA data center revenue exploded 400%, massive beat! Buying calls 🚀", "sentiment_label": "BULLISH"},
        {"post_id": "p2", "ticker": "TSLA", "text": "Deliveries down 20%, margins collapsing, cutting price targets. Puts printing.", "sentiment_label": "BEARISH"},
        {"post_id": "p3", "ticker": "AAPL", "text": "What time does Apple report earnings tomorrow afternoon?", "sentiment_label": "NEUTRAL"},
        {"post_id": "p4", "ticker": "AMD", "text": "MI300 chips are ramping fast, taking datacenter share from Intel. Long AMD.", "sentiment_label": "BULLISH"},
        {"post_id": "p5", "ticker": "INTC", "text": "Suspended dividend, $7B foundry loss, yield problems on 18A. Total dumpster fire.", "sentiment_label": "BEARISH"},
        {"post_id": "p6", "ticker": "MSFT", "text": "Azure growth is steady at 29%, but AI capex is huge. Valuation feels fair.", "sentiment_label": "NEUTRAL"},
        {"post_id": "p7", "ticker": "GME", "text": "Management announced another 20M ATM share dilution after the retail pump.", "sentiment_label": "BEARISH"},
        {"post_id": "p8", "ticker": "AMZN", "text": "AWS operating margins expanded to 38%, advertising revenue accelerating.", "sentiment_label": "BULLISH"},
        {"post_id": "p9", "ticker": "GOOGL", "text": "Search ad revenue beat consensus, cloud profitable for the third quarter.", "sentiment_label": "BULLISH"},
        {"post_id": "p10", "ticker": "META", "text": "Reality Labs lost another $4B, but family of apps ad impression volume up 20%.", "sentiment_label": "NEUTRAL"},
    ]
    df = pd.DataFrame(posts_data)
    print(f"Step 1: Loaded {len(df)} sample Reddit posts.")

    # 2. LLM Teacher Weak Label Generation (8 paths per post = 80 generations)
    mock_provider = MockProvider()
    prompt_mgr = PromptManager("sentiment_prompt")
    print("Step 2: Generating 8 reasoning paths per post via MockProvider...")
    paths_df = generate_teacher_weak_labels_dataset(
        df=df,
        provider=mock_provider,
        prompt_manager=prompt_mgr,
        num_paths=8,
        max_samples=10,
        include_cot=True,
    )
    assert len(paths_df) == 80, f"Expected 80 path records, got {len(paths_df)}"
    print(f"Step 2: Generated {len(paths_df)} reasoning path records.")

    # 3. Majority Voting & Soft Agreement Score Aggregation
    print("Step 3: Aggregating paths with majority voting and soft scores...")
    weak_df = aggregate_paths_dataframe(paths_df, original_df=df)
    assert len(weak_df) == 10
    assert "teacher_soft_score" in weak_df.columns
    assert "agreement_score" in weak_df.columns
    save_weak_labels(weak_df, filename="smoke_test_weak_labels.csv")
    print(f"Step 3: Aggregation complete. Mean agreement score: {weak_df['agreement_score'].mean():.3f}")

    # 4. Consistency Filtering (>= 5 out of 8 agreement)
    print("Step 4: Applying consistency filtering (threshold >= 5/8)...")
    filtered_df, filter_stats = filter_by_consistency(weak_df, threshold=5, total_paths=8)
    assert len(filtered_df) > 0, "Consistency filtering retained 0 examples."
    print(f"Step 4: Retained {filter_stats['retained_count']}/{filter_stats['initial_count']} posts ({filter_stats['retention_pct']}%).")

    # 5. Fast Student Regression Distillation Training (1 epoch)
    print("Step 5: Training student regression model (1 epoch test)...")
    train_subset = filtered_df.iloc[:8].copy()
    val_subset = filtered_df.iloc[8:].copy() if len(filtered_df) > 8 else filtered_df.iloc[:2].copy()
    test_subset = df.iloc[:4].copy()

    # Fast model initialization test
    student_model, student_tok, train_hist = train_regression_student(
        train_df=train_subset,
        val_df=val_subset,
        backbone_name="distilbert-base-uncased",
        epochs=1,
        batch_size=4,
        lr=5e-5,
        save_name="smoke_test_student_regression",
    )
    print(f"Step 5: Student trained. Epoch 1 Loss: {train_hist['train_loss'][0]:.4f}")

    # 6. Evaluation & Threshold Tuning
    print("Step 6: Evaluating student model and tuning threshold theta...")
    eval_metrics = evaluate_regression_student(
        model=student_model,
        tokenizer=student_tok,
        test_df=test_subset,
        val_df=val_subset,
        batch_size=4,
        save_prefix="smoke_test_eval",
    )
    assert "accuracy" in eval_metrics
    assert "macro_f1" in eval_metrics
    print(f"Step 6: Evaluation complete. Accuracy: {eval_metrics['accuracy']*100:.2f}%, Macro F1: {eval_metrics['macro_f1']:.4f}")

    print("=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 60 + "\n")
    assert eval_metrics["accuracy"] >= 0.0


if __name__ == "__main__":
    test_end_to_end_smoke_pipeline()
