"""
Create Annotation Set for Reddit Gold Evaluation.

This script creates the 100-post Reddit gold evaluation set from
emilpartow/reddit_finance_posts_sp500 that will be used for
held-out evaluation.

IMPORTANT:
- These 100 posts are NEVER used for teacher labeling or student training.
- They are kept completely separate from the distillation pool.
- The annotation template is created for human annotation.
- DO NOT FABRICATE human annotations.

Usage:
    python scripts/create_annotation_set.py [--output data/annotation_template.csv]
    python scripts/create_annotation_set.py --verify  # Check no leakage with distillation pool

Output:
    data/annotation_template.csv  -- 100 posts ready for human annotation
    docs/ANNOTATION_GUIDELINES.md -- Annotation instructions
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import load_reddit_hf_dataset, generate_sp500_dataset_stats
from src.data.preprocessing import combine_title_and_body, clean_text
from src.utils.common import DATA_RAW
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Configuration
ANNOTATION_SEED = 9999  # Different from distillation seed (42) to prevent leakage
ANNOTATION_SAMPLE_SIZE = 100
DISTILLATION_SEED = 42
DISTILLATION_MAX_SAMPLES = 800


def create_annotation_set(
    output_path: Path,
    sample_size: int = ANNOTATION_SAMPLE_SIZE,
    annotation_seed: int = ANNOTATION_SEED,
    distillation_seed: int = DISTILLATION_SEED,
    distillation_max_samples: int = DISTILLATION_MAX_SAMPLES,
    streaming: bool = True,
) -> pd.DataFrame:
    """
    Create reproducible 100-post annotation set from the sp500 dataset.

    Ensures ZERO overlap with the distillation pool.
    """
    logger.info("=" * 60)
    logger.info("CREATING REDDIT GOLD ANNOTATION SET")
    logger.info("=" * 60)

    logger.info(f"Loading sp500 dataset (streaming={streaming})...")
    df = load_reddit_hf_dataset(max_samples=1000, seed=annotation_seed, use_streaming=streaming)
    logger.info(f"Total available posts: {len(df)}")

    # Determine distillation pool post IDs (to exclude)
    distillation_rng = np.random.default_rng(distillation_seed)
    all_ids = df["post_id"].tolist()
    all_idx = list(range(len(df)))

    if distillation_max_samples < len(df):
        distillation_idx = sorted(
            distillation_rng.choice(len(df), size=distillation_max_samples, replace=False)
        )
        distillation_post_ids = set(df.iloc[distillation_idx]["post_id"].tolist())
    else:
        distillation_post_ids = set(df["post_id"].tolist())

    logger.info(f"Distillation pool size: {len(distillation_post_ids)} posts")

    # Select annotation candidates from posts NOT in distillation pool
    annotation_candidates = df[~df["post_id"].isin(distillation_post_ids)].copy()
    logger.info(f"Available non-distillation posts: {len(annotation_candidates)}")

    if len(annotation_candidates) < sample_size:
        logger.warning(
            f"Only {len(annotation_candidates)} non-distillation posts available, "
            f"requested {sample_size}. Using all available."
        )
        annotation_df = annotation_candidates.copy()
    else:
        # Sample reproducibly from non-distillation pool
        ann_rng = np.random.default_rng(annotation_seed)
        ann_idx = sorted(ann_rng.choice(len(annotation_candidates), size=sample_size, replace=False))
        annotation_df = annotation_candidates.iloc[ann_idx].copy().reset_index(drop=True)

    logger.info(f"Selected {len(annotation_df)} posts for annotation")

    # Verify zero overlap
    overlap = set(annotation_df["post_id"]) & distillation_post_ids
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} annotation posts found in distillation pool! "
            "This is a critical data integrity failure."
        )
    logger.info("[OK] LEAKAGE CHECK PASSED: Zero overlap between annotation set and distillation pool")

    # Build annotation template
    combined_texts = []
    for _, row in annotation_df.iterrows():
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        combined = combine_title_and_body(title, body) or clean_text(title)
        combined_texts.append(combined)

    template = pd.DataFrame({
        "post_id": annotation_df["post_id"].tolist(),
        "subreddit": annotation_df.get("subreddit", pd.Series(["unknown"] * len(annotation_df))).tolist(),
        "company": annotation_df.get("company", pd.Series([None] * len(annotation_df))).tolist(),
        "timestamp": annotation_df.get("timestamp", pd.Series([None] * len(annotation_df))).tolist(),
        "post_text": combined_texts,
        # Annotation columns — to be filled by human annotators
        "annotator_1": [""] * len(annotation_df),
        "annotator_2": [""] * len(annotation_df),
        "annotator_3": [""] * len(annotation_df),
        "consensus_label": [""] * len(annotation_df),
        "annotation_notes": [""] * len(annotation_df),
    })

    # Save annotation template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    logger.info(f"Saved annotation template to: {output_path}")

    # Save metadata
    meta = {
        "annotation_seed": annotation_seed,
        "distillation_seed": distillation_seed,
        "sample_size": len(annotation_df),
        "distillation_pool_size": len(distillation_post_ids),
        "overlap_count": len(overlap),
        "leakage_check_passed": len(overlap) == 0,
        "total_available_posts": len(df),
        "non_distillation_posts": len(annotation_candidates),
        "dataset_source": "emilpartow/reddit_finance_posts_sp500",
        "annotation_labels": ["positive", "neutral", "negative"],
        "post_ids_checksum": hashlib.sha256(
            json.dumps(sorted(annotation_df["post_id"].tolist())).encode()
        ).hexdigest(),
    }

    meta_path = output_path.parent / "annotation_set_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved annotation metadata to: {meta_path}")

    print("\n" + "=" * 60)
    print("ANNOTATION SET CREATED")
    print("=" * 60)
    print(f"Posts selected: {len(annotation_df)}")
    print(f"Distillation pool size: {len(distillation_post_ids)}")
    print(f"Leakage check: {'[PASSED]' if len(overlap) == 0 else '[FAILED]'}")
    print(f"Template saved to: {output_path}")
    print(f"\nNEXT STEPS:")
    print(f"1. Open {output_path} in a spreadsheet editor")
    print(f"2. Fill in 'annotator_1', 'annotator_2', 'annotator_3' columns")
    print(f"   Labels: positive | neutral | negative")
    print(f"3. Fill in 'consensus_label' based on majority agreement")
    print(f"4. Run: python scripts/validate_annotations.py")
    print(f"\nIMPORTANT: DO NOT FABRICATE ANNOTATIONS.")
    print(f"If genuine human annotations are not available, this evaluation")
    print(f"set must remain marked as BLOCKED in the final status report.")
    print("=" * 60)

    return annotation_df


def main():
    parser = argparse.ArgumentParser(description="Create Reddit gold annotation set")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotation_template.csv"),
        help="Output path for annotation template CSV",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=ANNOTATION_SAMPLE_SIZE,
        help=f"Number of posts to annotate (default: {ANNOTATION_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--annotation-seed",
        type=int,
        default=ANNOTATION_SEED,
        help=f"Random seed for annotation set selection (default: {ANNOTATION_SEED})",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming to load dataset (lower memory usage)",
    )
    args = parser.parse_args()

    annotation_df = create_annotation_set(
        output_path=args.output,
        sample_size=args.sample_size,
        annotation_seed=args.annotation_seed,
        streaming=args.streaming,
    )
    return annotation_df


if __name__ == "__main__":
    main()
