"""
End-to-End Smoke Test for the Paper Reproduction Pipeline.

Validates the complete pipeline without requiring any paid API calls.
Uses MockProvider and a 10-post synthetic dataset.

Pipeline tested:
  Hugging Face-like sample data
  → preprocessing
  → mock teacher (8 predictions per post)
  → majority vote
  → soft score
  → agreement filter (>= 5/8)
  → student mini-training (1 epoch)
  → evaluation

Usage:
    python scripts/smoke_test.py
    # OR
    pytest tests/test_smoke_pipeline.py -v

SYNTHETIC/MOCK DATA IS ALLOWED ONLY IN THIS SMOKE TEST.
Results from smoke tests are NOT research results.
"""

import sys
import io
import json
import time
from pathlib import Path

# Safe stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np


def run_smoke_test() -> dict:
    """
    Run the complete smoke test pipeline.

    Returns:
        Dictionary with smoke test results and pass/fail status.
    """
    print("=" * 65)
    print("SMOKE TEST: End-to-End Pipeline (No API Keys Required)")
    print("=" * 65)
    print("DATA: 10 synthetic Reddit posts (SMOKE TEST ONLY)")
    print("LLM:  MockProvider (deterministic, zero-cost)")
    print()

    results = {"steps": {}, "passed": True, "errors": []}
    t_start = time.time()

    # ── Step 1: Create test dataset ─────────────────────────────
    print("Step 1: Creating 10-post test dataset...")
    try:
        posts_data = [
            {"post_id": "smoke_p1", "ticker": "NVDA", "text": "NVDA data center revenue exploded 400%, massive beat! Buying calls", "sentiment_label": "BULLISH"},
            {"post_id": "smoke_p2", "ticker": "TSLA", "text": "Deliveries down 20%, margins collapsing, cutting price targets. Puts printing.", "sentiment_label": "BEARISH"},
            {"post_id": "smoke_p3", "ticker": "AAPL", "text": "What time does Apple report earnings tomorrow afternoon?", "sentiment_label": "NEUTRAL"},
            {"post_id": "smoke_p4", "ticker": "AMD", "text": "MI300 chips are ramping fast, taking datacenter share from Intel. Long AMD.", "sentiment_label": "BULLISH"},
            {"post_id": "smoke_p5", "ticker": "INTC", "text": "Suspended dividend, $7B foundry loss, yield problems on 18A. Total dumpster fire.", "sentiment_label": "BEARISH"},
            {"post_id": "smoke_p6", "ticker": "MSFT", "text": "Azure growth is steady at 29%, but AI capex is huge. Valuation feels fair.", "sentiment_label": "NEUTRAL"},
            {"post_id": "smoke_p7", "ticker": "GME", "text": "Management announced another 20M ATM share dilution after the retail pump.", "sentiment_label": "BEARISH"},
            {"post_id": "smoke_p8", "ticker": "AMZN", "text": "AWS operating margins expanded to 38%, advertising revenue accelerating.", "sentiment_label": "BULLISH"},
            {"post_id": "smoke_p9", "ticker": "GOOGL", "text": "Search ad revenue beat consensus, cloud profitable for the third quarter.", "sentiment_label": "BULLISH"},
            {"post_id": "smoke_p10", "ticker": "META", "text": "Reality Labs lost another $4B, but family of apps ad impression volume up 20%.", "sentiment_label": "NEUTRAL"},
        ]
        df = pd.DataFrame(posts_data)
        df["data_source"] = "SYNTHETIC"  # smoke test uses synthetic data
        assert len(df) == 10
        results["steps"]["create_dataset"] = {"status": "PASS", "n_posts": len(df)}
        print(f"  [OK] Created {len(df)} test posts")
    except Exception as e:
        results["steps"]["create_dataset"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 1 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")

    # ── Step 2: Preprocessing ────────────────────────────────────
    print("Step 2: Text preprocessing...")
    try:
        from src.data.preprocessing import combine_title_and_body, clean_text
        assert df["text"].notna().all()
        assert df["text"].str.len().min() > 5
        results["steps"]["preprocessing"] = {"status": "PASS", "n_valid": int((df["text"].str.len() > 5).sum())}
        print(f"  [OK] All {len(df)} posts have valid text")
    except Exception as e:
        results["steps"]["preprocessing"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 2 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")

    # ── Step 3: LLM Teacher Weak Labeling (8 paths, MockProvider) ──
    print("Step 3: Mock teacher labeling (8 paths x 10 posts = 80 generations)...")
    try:
        from src.llm.client import MockProvider
        from src.llm.prompts import PromptManager
        from src.llm.labeling import generate_teacher_weak_labels_dataset

        mock_provider = MockProvider()
        prompt_mgr = PromptManager("sentiment_prompt")
        paths_df = generate_teacher_weak_labels_dataset(
            df=df,
            provider=mock_provider,
            prompt_manager=prompt_mgr,
            num_paths=8,
            max_samples=10,
            include_cot=True,
        )
        assert len(paths_df) == 80, f"Expected 80 path records, got {len(paths_df)}"
        assert "post_id" in paths_df.columns
        assert "label" in paths_df.columns
        assert "reasoning_summary" in paths_df.columns
        assert set(paths_df["label"].unique()).issubset({"positive", "neutral", "negative"})
        results["steps"]["teacher_labeling"] = {
            "status": "PASS",
            "n_paths": len(paths_df),
            "label_distribution": paths_df["label"].value_counts().to_dict(),
        }
        print(f"  [OK] Generated {len(paths_df)} reasoning path records")
        print(f"       Label distribution: {paths_df['label'].value_counts().to_dict()}")
    except Exception as e:
        results["steps"]["teacher_labeling"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 3 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")
        paths_df = None

    # ── Step 4: Majority Voting and Soft Score ───────────────────
    print("Step 4: Majority voting and soft score aggregation...")
    try:
        from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels

        weak_df = aggregate_paths_dataframe(paths_df, original_df=df)
        assert len(weak_df) == 10, f"Expected 10 aggregated records, got {len(weak_df)}"
        assert "teacher_soft_score" in weak_df.columns
        assert "agreement_score" in weak_df.columns
        assert "dominant_label" in weak_df.columns
        assert weak_df["agreement_score"].between(0.0, 1.0).all()
        assert weak_df["teacher_soft_score"].between(-1.0, 1.0).all()

        save_weak_labels(weak_df, filename="smoke_test_weak_labels.csv")

        results["steps"]["aggregation"] = {
            "status": "PASS",
            "n_posts": len(weak_df),
            "mean_agreement": round(float(weak_df["agreement_score"].mean()), 4),
            "soft_score_range": [
                round(float(weak_df["teacher_soft_score"].min()), 4),
                round(float(weak_df["teacher_soft_score"].max()), 4),
            ],
        }
        print(f"  [OK] Aggregated {len(weak_df)} posts | Mean agreement: {weak_df['agreement_score'].mean():.3f}")
    except Exception as e:
        results["steps"]["aggregation"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 4 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")
        weak_df = None

    # ── Step 5: Agreement Filter (>= 5/8) ───────────────────────
    print("Step 5: Consistency filtering (agreement >= 5/8)...")
    try:
        from src.distillation.dataset import filter_by_consistency

        filtered_df, filter_stats = filter_by_consistency(weak_df, threshold=5, total_paths=8)
        assert len(filtered_df) > 0, "Filtering removed ALL examples — check MockProvider consistency"
        assert "teacher_soft_score" in filtered_df.columns

        results["steps"]["agreement_filter"] = {
            "status": "PASS",
            "initial": filter_stats["initial_count"],
            "retained": filter_stats["retained_count"],
            "retention_pct": filter_stats["retention_pct"],
        }
        print(f"  [OK] Retained {filter_stats['retained_count']}/{filter_stats['initial_count']} "
              f"posts ({filter_stats['retention_pct']}%) with agreement >= 5/8")
    except Exception as e:
        results["steps"]["agreement_filter"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 5 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")
        filtered_df = None

    # ── Step 6: Student Mini-Training (1 epoch) ──────────────────
    print("Step 6: Student regression training (1 epoch, distilbert-base-uncased)...")
    try:
        import torch
        from src.distillation.train import train_regression_student

        train_subset = filtered_df.iloc[:min(8, len(filtered_df))].copy()
        val_subset = filtered_df.iloc[-min(2, len(filtered_df)):].copy()

        student_model, student_tok, train_hist = train_regression_student(
            train_df=train_subset,
            val_df=val_subset,
            backbone_name="distilbert-base-uncased",
            epochs=1,
            batch_size=2,
            lr=5e-5,
            max_length=128,
            save_name="smoke_test_student_regression",
        )
        assert len(train_hist["train_loss"]) == 1
        assert train_hist["train_loss"][0] > 0

        results["steps"]["student_training"] = {
            "status": "PASS",
            "epochs": 1,
            "final_train_loss": round(float(train_hist["train_loss"][0]), 6),
            "final_val_mse": round(float(train_hist["val_mse"][0]), 6),
        }
        print(f"  [OK] Student trained | Train Loss: {train_hist['train_loss'][0]:.4f} | Val MSE: {train_hist['val_mse'][0]:.4f}")
    except Exception as e:
        results["steps"]["student_training"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 6 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")
        student_model, student_tok = None, None

    # ── Step 7: Evaluation ───────────────────────────────────────
    print("Step 7: Student model evaluation...")
    try:
        from src.distillation.evaluate import evaluate_regression_student

        test_subset = df.iloc[:4].copy()
        if "teacher_soft_score" not in test_subset.columns:
            test_subset["teacher_soft_score"] = 0.0

        eval_metrics = evaluate_regression_student(
            model=student_model,
            tokenizer=student_tok,
            test_df=test_subset,
            val_df=val_subset,
            batch_size=2,
            save_prefix="smoke_test_eval",
        )
        assert "accuracy" in eval_metrics
        assert "macro_f1" in eval_metrics
        assert 0.0 <= eval_metrics["accuracy"] <= 1.0

        results["steps"]["evaluation"] = {
            "status": "PASS",
            "accuracy": round(float(eval_metrics["accuracy"]), 4),
            "macro_f1": round(float(eval_metrics["macro_f1"]), 4),
        }
        print(f"  [OK] Evaluation complete | Accuracy: {eval_metrics['accuracy']*100:.1f}% | Macro F1: {eval_metrics['macro_f1']:.4f}")
    except Exception as e:
        results["steps"]["evaluation"] = {"status": "FAIL", "error": str(e)}
        results["errors"].append(f"Step 7 FAILED: {e}")
        results["passed"] = False
        print(f"  [FAIL] FAILED: {e}")

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results["elapsed_seconds"] = round(elapsed, 2)
    results["n_steps_passed"] = sum(1 for s in results["steps"].values() if s.get("status") == "PASS")
    results["n_steps_total"] = len(results["steps"])

    print()
    print("=" * 65)
    if results["passed"] and not results["errors"]:
        print("[OK] ALL SMOKE TEST STEPS PASSED")
    else:
        print("[FAIL] SMOKE TEST COMPLETED WITH ERRORS")
    print(f"   Steps: {results['n_steps_passed']}/{results['n_steps_total']} passed")
    print(f"   Time:  {elapsed:.1f}s")
    if results["errors"]:
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  [FAIL] {e}")
    print()
    print("IMPORTANT: This test used SYNTHETIC data and a MockProvider.")
    print("Results from smoke tests are NOT valid research results.")
    print("=" * 65)

    # Save results
    out_path = Path("outputs/smoke_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSmoke test results saved to: {out_path}")

    return results


if __name__ == "__main__":
    results = run_smoke_test()
    sys.exit(0 if results["passed"] else 1)
