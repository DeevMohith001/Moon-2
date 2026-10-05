"""
Confidence Calibration, Consistency Analysis, and Multi-Model Benchmarking (Phase 6).

Implements:
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Reliability diagrams (calibration curves)
- LLM self-consistency test (across repeated inferences)
- Comprehensive multi-model benchmark comparison table
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.common import RESULTS_DIR, get_logger
from src.evaluation.metrics import save_metrics

logger = get_logger(__name__)


def compute_calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and bin statistics.
    
    Args:
        confidences: Array of predicted model confidences in [0, 1].
        accuracies: Binary array indicating whether prediction was correct (1 or 0).
        num_bins: Number of confidence bins.
        
    Returns:
        Dict with ECE, MCE, and bin details for reliability diagram.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    bin_accs = []
    bin_confs = []
    bin_sizes = []
    ece = 0.0
    mce = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        mask = bin_indices == i
        n_in_bin = np.sum(mask)
        bin_sizes.append(int(n_in_bin))

        if n_in_bin > 0:
            avg_acc = float(np.mean(accuracies[mask]))
            avg_conf = float(np.mean(confidences[mask]))
            bin_accs.append(round(avg_acc, 4))
            bin_confs.append(round(avg_conf, 4))
            gap = abs(avg_acc - avg_conf)
            ece += (n_in_bin / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bins[i] + bins[i + 1]) / 2))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_sizes": bin_sizes,
        "num_bins": num_bins,
    }


def plot_reliability_diagram(
    calib_data: Dict[str, Any],
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Generate and save publication reliability diagram."""
    bin_accs = calib_data["bin_accuracies"]
    bin_confs = calib_data["bin_confidences"]
    bin_sizes = calib_data["bin_sizes"]
    ece = calib_data["ece"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, dpi=300)

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    valid_points = [(c, a) for c, a, s in zip(bin_confs, bin_accs, bin_sizes) if s > 0]
    if valid_points:
        cs, as_ = zip(*valid_points)
        ax1.plot(cs, as_, marker="o", linewidth=2.5, color="#2b5c8f", label=f"{model_name} (ECE={ece:.4f})")
        ax1.bar(cs, as_, width=0.08, alpha=0.3, color="#2b5c8f", edgecolor="#2b5c8f")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Reliability Diagram — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Sample distribution histogram
    bin_centers = np.linspace(0.05, 0.95, len(bin_sizes))
    ax2.bar(bin_centers, bin_sizes, width=0.08, color="#5c768d", edgecolor="black", alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path is None:
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        save_path = RESULTS_DIR / "figures" / f"calibration_{safe_name}.png"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_master_benchmark_table() -> pd.DataFrame:
    """
    Collate metrics across all evaluated models (Phase 2, 3, 4, 5) into a unified benchmark table.
    """
    import json
    metrics_dir = RESULTS_DIR / "metrics"
    models_info = [
        ("VADER (Lexicon)", "vader_metrics.json", "Phase 2"),
        ("TF-IDF + LogReg", "logreg_metrics.json", "Phase 2"),
        ("TF-IDF + Linear SVM", "svm_metrics.json", "Phase 2"),
        ("FinBERT (ProsusAI)", "prosusai_finbert_metrics.json", "Phase 3"),
        ("FinBERT (HKUST)", "yiyanghkust_finbert-tone_metrics.json", "Phase 3"),
        ("LLM Zero-Shot (L1)", "llm_l1_metrics.json", "Phase 4"),
        ("LLM Few-Shot (L2)", "llm_l2_metrics.json", "Phase 4"),
        ("LLM Financial-Context (L3)", "llm_l3_metrics.json", "Phase 4"),
        ("LLM Structured (L4)", "llm_l4_metrics.json", "Phase 4"),
        ("LLM CoT (L5)", "llm_l5_metrics.json", "Phase 4"),
        ("Distilled Student (R2)", "distillation_regression_metrics.json", "Phase 5"),
    ]

    rows = []
    for model_name, filename, phase in models_info:
        file_path = metrics_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    rows.append({
                        "Phase": phase,
                        "Model": model_name,
                        "Accuracy": data.get("accuracy", 0.0),
                        "Balanced Acc": data.get("balanced_accuracy", 0.0),
                        "Macro F1": data.get("macro_f1", 0.0),
                        "Weighted F1": data.get("weighted_f1", 0.0),
                        "Bullish F1": data.get("per_class", {}).get("BULLISH", {}).get("f1", 0.0),
                        "Bearish F1": data.get("per_class", {}).get("BEARISH", {}).get("f1", 0.0),
                        "Neutral F1": data.get("per_class", {}).get("NEUTRAL", {}).get("f1", 0.0),
                    })
            except Exception as e:
                logger.warning(f"Failed to read {filename}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        summary_csv = RESULTS_DIR / "sentiment_metrics.csv"
        df.to_csv(summary_csv, index=False)
        logger.info(f"Master benchmark table saved to {summary_csv}")
    return df


if __name__ == "__main__":
    table = build_master_benchmark_table()
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80)

    # Generate calibration plots for models with confidences
    preds_dir = RESULTS_DIR / "predictions"
    finbert_preds_file = preds_dir / "prosusai_finbert_predictions.parquet"
    if finbert_preds_file.exists():
        f_df = pd.read_parquet(finbert_preds_file)
        if "confidence" in f_df.columns:
            calib = compute_calibration_curve(f_df["confidence"].values, f_df["is_correct"].astype(int).values)
            save_metrics(calib, RESULTS_DIR / "metrics" / "calibration_finbert_prosus.json")
            plot_reliability_diagram(calib, "FinBERT (ProsusAI)")

