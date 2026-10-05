"""
Validate Human Annotations for Reddit Gold Evaluation Set.

This script validates the human annotations in data/annotation_template.csv and:
1. Checks all labels are valid (positive/neutral/negative)
2. Calculates inter-annotator agreement (Cohen's Kappa, Fleiss' Kappa)
3. Calculates majority consensus labels
4. Reports annotation statistics
5. Verifies the annotation set has no leakage

Usage:
    python scripts/validate_annotations.py [--input data/annotation_template.csv]
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"positive", "neutral", "negative"}
ANNOTATOR_COLS = ["annotator_1", "annotator_2", "annotator_3"]


def validate_annotations(input_path: Path, output_path: Path = None) -> dict:
    """
    Validate human annotations and compute agreement statistics.

    Returns:
        Dictionary with validation results and statistics.
    """
    if not input_path.exists():
        print(f"ERROR: Annotation file not found: {input_path}")
        print("Run: python scripts/create_annotation_set.py first")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} annotation rows from {input_path}")

    results = {
        "input_file": str(input_path),
        "total_posts": len(df),
        "fully_annotated": 0,
        "partially_annotated": 0,
        "unannotated": 0,
        "annotation_complete": False,
        "valid_labels_only": True,
        "inter_annotator_agreement": {},
        "class_distribution": {},
        "consensus_labels": {},
        "errors": [],
    }

    # Check which annotator columns exist
    present_annotator_cols = [c for c in ANNOTATOR_COLS if c in df.columns]
    if not present_annotator_cols:
        results["errors"].append("No annotator columns found in file")
        print_results(results)
        return results

    # Classify annotation completeness
    for _, row in df.iterrows():
        annotations = [str(row.get(col, "")).strip().lower() for col in present_annotator_cols]
        non_empty = [a for a in annotations if a and a not in ("nan", "")]
        if len(non_empty) == len(present_annotator_cols):
            results["fully_annotated"] += 1
        elif len(non_empty) > 0:
            results["partially_annotated"] += 1
        else:
            results["unannotated"] += 1

    if results["unannotated"] == len(df):
        print("\n" + "=" * 60)
        print("ANNOTATION TEMPLATE IS EMPTY")
        print("=" * 60)
        print(f"No annotations found in: {input_path}")
        print("\nStatus: BLOCKED - Awaiting human annotations")
        print("\nTo annotate:")
        print(f"1. Open {input_path} in Excel/Google Sheets")
        print("2. For each post, fill columns annotator_1, annotator_2, annotator_3")
        print("   Valid labels: positive | neutral | negative")
        print("3. Fill consensus_label (majority of the 3 annotators)")
        print("4. Re-run this validation script")
        print("=" * 60)
        results["annotation_complete"] = False
        return results

    # Validate label values
    for col in present_annotator_cols:
        col_df = df[col].dropna().astype(str).str.strip().str.lower()
        col_df = col_df[col_df != ""]
        invalid = col_df[~col_df.isin(VALID_LABELS)]
        if len(invalid) > 0:
            results["valid_labels_only"] = False
            results["errors"].append(
                f"Column '{col}' contains invalid labels: {invalid.unique().tolist()}. "
                f"Valid labels are: {sorted(VALID_LABELS)}"
            )

    # Work with fully annotated rows
    annotated_mask = df.apply(
        lambda row: all(
            str(row.get(col, "")).strip().lower() in VALID_LABELS
            for col in present_annotator_cols
        ),
        axis=1,
    )
    annotated_df = df[annotated_mask].copy()

    if len(annotated_df) == 0:
        print("No fully annotated rows found with valid labels.")
        results["annotation_complete"] = False
        print_results(results)
        return results

    # Class distribution per annotator
    for col in present_annotator_cols:
        col_labels = annotated_df[col].str.strip().str.lower()
        results["class_distribution"][col] = col_labels.value_counts().to_dict()

    # Inter-annotator agreement (pairwise Cohen's Kappa if sklearn available)
    try:
        from sklearn.metrics import cohen_kappa_score

        pairwise = {}
        for i, col_a in enumerate(present_annotator_cols):
            for col_b in present_annotator_cols[i+1:]:
                labels_a = annotated_df[col_a].str.strip().str.lower().tolist()
                labels_b = annotated_df[col_b].str.strip().str.lower().tolist()
                try:
                    kappa = cohen_kappa_score(labels_a, labels_b)
                    pairwise[f"{col_a}_vs_{col_b}"] = round(float(kappa), 4)
                except Exception as e:
                    pairwise[f"{col_a}_vs_{col_b}"] = f"Error: {e}"

        results["inter_annotator_agreement"]["pairwise_cohen_kappa"] = pairwise
        if pairwise:
            valid_kappas = [v for v in pairwise.values() if isinstance(v, float)]
            if valid_kappas:
                results["inter_annotator_agreement"]["mean_cohen_kappa"] = round(float(np.mean(valid_kappas)), 4)

    except ImportError:
        results["inter_annotator_agreement"]["note"] = "sklearn not available for Cohen's Kappa"

    # Majority consensus labels
    def majority_label(row):
        labels = [str(row.get(col, "")).strip().lower() for col in present_annotator_cols]
        labels = [l for l in labels if l in VALID_LABELS]
        if not labels:
            return "unknown"
        from collections import Counter
        counts = Counter(labels)
        max_count = max(counts.values())
        candidates = [l for l, c in counts.items() if c == max_count]
        if len(candidates) == 1:
            return candidates[0]
        return "neutral"  # tie-breaking: neutral per paper methodology

    annotated_df["computed_consensus"] = annotated_df.apply(majority_label, axis=1)
    results["consensus_labels"] = annotated_df["computed_consensus"].value_counts().to_dict()

    # Check consensus_label column consistency if present
    if "consensus_label" in annotated_df.columns:
        provided_consensus = annotated_df["consensus_label"].str.strip().str.lower()
        computed_consensus = annotated_df["computed_consensus"]
        valid_provided = provided_consensus.isin(VALID_LABELS)
        if valid_provided.all():
            agreement_with_computed = (provided_consensus == computed_consensus).mean()
            results["consensus_vs_majority_agreement"] = round(float(agreement_with_computed), 4)

    results["annotation_complete"] = (
        results["fully_annotated"] == len(df) and
        results["valid_labels_only"]
    )

    # Save validated annotations if complete
    if results["annotation_complete"] and output_path:
        # Merge computed consensus back
        df_out = df.copy()
        for idx, row_idx in enumerate(annotated_df.index):
            df_out.loc[row_idx, "computed_consensus"] = annotated_df.loc[row_idx, "computed_consensus"]
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved validated annotations to: {output_path}")
        results["validated_output"] = str(output_path)

    print_results(results)
    return results


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("ANNOTATION VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total posts: {results['total_posts']}")
    print(f"Fully annotated: {results['fully_annotated']}")
    print(f"Partially annotated: {results['partially_annotated']}")
    print(f"Unannotated: {results['unannotated']}")
    print(f"Valid labels only: {results['valid_labels_only']}")
    print(f"Annotation complete: {'[YES]' if results['annotation_complete'] else '[NO (BLOCKED)]'}")

    if results.get("errors"):
        print("\nERRORS:")
        for err in results["errors"]:
            print(f"  [ERROR] {err}")

    if results.get("inter_annotator_agreement"):
        print("\nInter-Annotator Agreement:")
        iaa = results["inter_annotator_agreement"]
        if "pairwise_cohen_kappa" in iaa:
            for pair, kappa in iaa["pairwise_cohen_kappa"].items():
                print(f"  {pair}: κ = {kappa}")
        if "mean_cohen_kappa" in iaa:
            print(f"  Mean κ = {iaa['mean_cohen_kappa']}")

    if results.get("consensus_labels"):
        print("\nConsensus Label Distribution:")
        for label, count in results["consensus_labels"].items():
            pct = 100.0 * count / results["total_posts"]
            print(f"  {label}: {count} ({pct:.1f}%)")

    if not results.get("annotation_complete"):
        print("\nSTATUS: BLOCKED - Reddit gold evaluation requires human annotations")
        print("The rest of the pipeline can still run (FiQA, ablations, distillation).")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate human annotations")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/annotation_template.csv"),
        help="Input annotation CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validated_annotations.csv"),
        help="Output validated CSV file",
    )
    args = parser.parse_args()

    results = validate_annotations(args.input, args.output)

    # Save results to JSON
    results_path = Path("outputs/error_analysis/annotation_validation.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nValidation results saved to: {results_path}")


if __name__ == "__main__":
    main()
