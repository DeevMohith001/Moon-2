"""
Categorized Error Analysis and Model Disagreement Diagnostics (Section 19).

Taxonomy of Social Media & Financial Error Sources:
1. SARCASM_MEMES: Sarcastic memes (e.g., 🤡, "literally cannot go tits up", "guh", "paper hands")
2. CONTRADICTORY_ARGUMENTS: Explicit conflicting arguments ("revenue up but margins collapsing")
3. ADVANCED_INVESTING_TERMINOLOGY: Options/derivatives concepts (IV crush, gamma squeeze, delta hedging, iron condors)
4. SLANG_JARGON: Subreddit-specific slang ("tendies", "bagholder", "loss porn", "to the moon")
5. MULTIPLE_COMPANIES_TICKERS: Multiple equities mentioned with opposing outlooks
6. LONG_DUE_DILIGENCE_POSTS: Very long posts (> 1000 characters) with shifting narrative
7. TICKER_AMBIGUITY: Words matching English dictionary terms (e.g. IT, FOR, BE, ALL)
8. SUBTLE_IMPLICIT_SENTIMENT: Ambiguous or subtle tone without explicit directional indicators
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.common import OUTPUTS_DIR, RESULTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

ERROR_PATTERNS = {
    "SARCASM_MEMES": [
        r"\bclown\b", r"🤡", r"\bguh\b", r"\btits up\b", r"\bpaper hands\b",
        r"\bdiamond hands\b", r"\bbagholder\b", r"\bto the moon\b", r"\bstonks\b",
        r"\bloss porn\b", r"\bwife's boyfriend\b", r"\byolo\b",
    ],
    "CONTRADICTORY_ARGUMENTS": [
        r"\bbeat earnings but\b", r"\bgood revenue however\b", r"\bon the other hand\b",
        r"\blong term bullish but short term\b", r"\bshort term.*long term\b",
        r"\balbeit\b", r"\bnonetheless\b", r"\bmixed feelings\b",
    ],
    "ADVANCED_INVESTING_TERMINOLOGY": [
        r"\biv crush\b", r"\bgamma squeeze\b", r"\bmax pain\b", r"\btheta\b",
        r"\bdelta neutral\b", r"\bleaps\b", r"\biron condor\b", r"\bcovered call\b",
        r"\bshort squeeze\b", r"\bliquidity sweep\b", r"\bmacro headwind\b",
    ],
    "SLANG_JARGON": [
        r"\btendies\b", r"\bdrill\b", r"\bdrilling\b", r"\btank\b", r"\btanking\b",
        r"\bprinting\b", r"\bape\b", r"\bapes\b", r"\brug pull\b",
    ],
    "MULTIPLE_COMPANIES_TICKERS": [
        r"\bvs\b", r"\bcompared to\b", r"\bbetter than\b", r"\binstead of\b",
        r"\brotat(?:e|ing|ion)\b",
    ],
}


def categorize_error(text: str) -> str:
    """
    Determine the most prominent linguistic/financial challenge in a post.
    """
    lower = text.lower()

    if len(text) > 1000:
        return "LONG_DUE_DILIGENCE_POSTS"

    for category, patterns in ERROR_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower):
                return category

    return "SUBTLE_IMPLICIT_SENTIMENT"


def analyze_model_errors(
    test_df: pd.DataFrame,
    preds_dict: Dict[str, List[str]],
    text_col: str = "clean_text",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Analyze model disagreement and failure cases across multiple models.
    preds_dict: map of model_name -> list of predictions on test_df
    """
    actuals = test_df[label_col].tolist()
    texts = test_df[text_col if text_col in test_df.columns else "text"].tolist()

    records = []
    for i, (text, true_y) in enumerate(zip(texts, actuals)):
        category = categorize_error(text)
        rec = {
            "post_id": test_df.iloc[i].get("post_id", f"post_{i}"),
            "text": text[:300] + ("..." if len(text) > 300 else ""),
            "actual_label": true_y,
            "error_category": category,
        }

        # Check each model
        num_errors = 0
        for m_name, preds in preds_dict.items():
            pred_y = preds[i]
            is_err = str(pred_y).strip().upper() != str(true_y).strip().upper()
            rec[f"{m_name}_pred"] = pred_y
            rec[f"{m_name}_error"] = is_err
            if is_err:
                num_errors += 1

        rec["total_models_failing"] = num_errors
        records.append(rec)

    err_df = pd.DataFrame(records)

    # Persist detailed diagnostics
    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        err_df.to_csv(target_dir / "error_analysis.csv", index=False)

    logger.info(
        f"Error Analysis complete for {len(err_df)} posts. Category breakdown:\n"
        + err_df["error_category"].value_counts().to_string()
    )
    return err_df
