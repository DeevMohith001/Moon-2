"""
Strict LLM Output Parser and Sentiment Normalization.

Enforces output structure:
{
    "reasoning_summary": "...",
    "sentiment": "positive|neutral|negative"
}
Ensures no arbitrary labels are accepted and malformed outputs are handled strictly.
"""

import re
import json
from typing import Dict, Any, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

# Canonical mapping to paper display classes
SENTIMENT_TO_DISPLAY = {
    "positive": "BULLISH",
    "neutral": "NEUTRAL",
    "negative": "BEARISH",
}

DISPLAY_TO_NORMALIZED = {
    "BULLISH": "positive",
    "NEUTRAL": "neutral",
    "BEARISH": "negative",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
}


def normalize_sentiment_label(raw_label: str) -> str:
    """
    Normalize raw sentiment string to canonical lowercase: 'positive', 'neutral', 'negative'.
    Raises ValueError if label cannot be mapped.
    """
    cleaned = raw_label.strip().lower()

    if cleaned in VALID_SENTIMENTS:
        return cleaned

    # Check common synonyms
    if cleaned in ["bullish", "bull", "buy", "long", "upside"]:
        return "positive"
    if cleaned in ["bearish", "bear", "sell", "short", "downside"]:
        return "negative"
    if cleaned in ["neutral", "hold", "uncertain", "indeterminate", "mixed"]:
        return "neutral"

    raise ValueError(f"Invalid sentiment label '{raw_label}'. Allowed: {VALID_SENTIMENTS}")


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Strict parser for LLM generation response.
    Attempts JSON parsing, cleans markdown fences if present, validates keys and normalized label.
    """
    if not response_text or not isinstance(response_text, str):
        raise ValueError("Empty or non-string response received from LLM.")

    text = response_text.strip()

    # Strip markdown code fences if model wrapped response in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    parsed_json = None

    # Try direct JSON parse
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON object substring via regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if parsed_json and isinstance(parsed_json, dict):
        raw_sentiment = parsed_json.get("sentiment")
        if not raw_sentiment:
            raise ValueError("Parsed JSON missing required key 'sentiment'.")

        normalized = normalize_sentiment_label(str(raw_sentiment))
        reasoning = str(parsed_json.get("reasoning_summary", "")).strip()
        if not reasoning:
            reasoning = "No explicit reasoning provided in response."

        return {
            "sentiment": normalized,
            "display_label": SENTIMENT_TO_DISPLAY[normalized],
            "reasoning_summary": reasoning,
            "raw_response": response_text,
        }

    # Fallback heuristic parser for plain-text CoT formats
    # E.g.: "REASONING: ... \nSENTIMENT: Bullish"
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\nSENTIMENT|\n*$)", text, re.DOTALL | re.IGNORECASE)
    sentiment_match = re.search(r"SENTIMENT:\s*([A-Za-z]+)", text, re.IGNORECASE)

    if sentiment_match:
        norm_label = normalize_sentiment_label(sentiment_match.group(1))
        reason_text = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "Reasoning extracted from structured text."
        )
        return {
            "sentiment": norm_label,
            "display_label": SENTIMENT_TO_DISPLAY[norm_label],
            "reasoning_summary": reason_text,
            "raw_response": response_text,
        }

    # If completely unparseable, log failure and raise
    logger.error(f"Malformed LLM output could not be parsed: {response_text[:200]}...")
    raise ValueError(f"LLM output could not be parsed into required schema: {response_text[:100]}")
