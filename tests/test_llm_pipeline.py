"""
Unit tests for Teacher LLM pipeline:
- Prompt construction & 6-demonstration formatting
- Strict parser & sentiment label normalization
- Multi-path generation & Mock provider
- Majority voting & tie-breaking
- Soft agreement score calculation
- Consistency filtering
- Student dataset creation
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import MockProvider, get_llm_provider
from src.llm.prompts import PromptManager
from src.llm.parser import parse_llm_response, normalize_sentiment_label
from src.llm.labeling import generate_reasoning_paths_for_post
from src.llm.aggregation import aggregate_post_paths, aggregate_paths_dataframe
from src.distillation.dataset import filter_by_consistency, TextRegressionDataset, TextClassificationDataset


def test_prompt_construction():
    """Verify prompt manager loads 6 demonstrations and renders valid prompt."""
    pm = PromptManager("sentiment_prompt")
    assert len(pm.demonstrations) == 6
    prompt = pm.build_prompt("TSLA deliveries reached new record high", ticker="TSLA", include_cot=True)
    assert "TSLA" in prompt
    assert "Example 1:" in prompt
    assert "reasoning_summary" in prompt


def test_demonstration_shuffle():
    """Verify demonstration order can be deterministically shuffled for ablations."""
    pm = PromptManager("sentiment_prompt")
    d1 = pm.format_demonstrations(shuffle_seed=42)
    d2 = pm.format_demonstrations(shuffle_seed=101)
    assert len(d1) > 0
    assert len(d2) > 0


def test_strict_llm_parser_valid_json():
    """Verify valid JSON is parsed and normalized correctly."""
    valid_json = '{"reasoning_summary": "Strong server shipments", "sentiment": "bullish"}'
    parsed = parse_llm_response(valid_json)
    assert parsed["sentiment"] == "positive"
    assert parsed["display_label"] == "BULLISH"
    assert "Strong server shipments" in parsed["reasoning_summary"]


def test_strict_llm_parser_markdown_fence():
    """Verify markdown code fences are stripped cleanly."""
    fenced = '```json\n{"reasoning_summary": "Revenue miss and guide cut", "sentiment": "bearish"}\n```'
    parsed = parse_llm_response(fenced)
    assert parsed["sentiment"] == "negative"
    assert parsed["display_label"] == "BEARISH"


def test_strict_llm_parser_invalid_label_rejection():
    """Verify unmapped arbitrary labels raise ValueError."""
    bad_json = '{"reasoning_summary": "Something", "sentiment": "crypto_moon_laser_eyes"}'
    with pytest.raises(ValueError):
        parse_llm_response(bad_json)


def test_mock_llm_provider():
    """Verify deterministic mock provider generates valid parseable responses."""
    mock = MockProvider()
    resp = mock.generate("Analysis for NVDA: strong earnings growth and call options buying")
    parsed = parse_llm_response(resp)
    assert parsed["sentiment"] == "positive"
    assert len(parsed["reasoning_summary"]) > 0


def test_multi_path_generation_and_schema():
    """Verify 8 reasoning paths are generated with the exact required schema."""
    provider = MockProvider()
    pm = PromptManager("sentiment_prompt")
    paths = generate_reasoning_paths_for_post(
        post_id="post_test_1",
        text="Massive datacenter earnings beat! Buying calls 🚀",
        ticker="NVDA",
        provider=provider,
        prompt_manager=pm,
        num_paths=8,
        include_cot=True,
    )
    assert len(paths) == 8
    for p in paths:
        assert p["post_id"] == "post_test_1"
        assert "path_id" in p
        assert "reasoning_summary" in p
        assert p["label"] in ["positive", "neutral", "negative"]
        assert "confidence" in p
        assert "timestamp" in p


def test_mock_provider_sampling_path_variation():
    """Verify the mock provider honors the same sampling marker emitted by labeling."""
    provider = MockProvider()
    common_prompt = "NVDA strong growth beat and calls buying"

    path_1 = parse_llm_response(provider.generate(f"{common_prompt}\n\n[Sampling Path 1/8]"))
    path_7 = parse_llm_response(provider.generate(f"{common_prompt}\n\n[Sampling Path 7/8]"))

    assert path_1["sentiment"] == "positive"
    assert path_7["sentiment"] == "neutral"


def test_majority_voting_and_soft_score():
    """Verify majority voting, agreement score, and soft continuous target math."""
    mock_paths = pd.DataFrame([
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "positive", "confidence": 0.9, "reasoning_summary": "bullish reason"},
        {"post_id": "p1", "label": "neutral", "confidence": 0.7, "reasoning_summary": "neutral reason"},
        {"post_id": "p1", "label": "neutral", "confidence": 0.7, "reasoning_summary": "neutral reason"},
    ])
    agg = aggregate_post_paths(mock_paths)
    assert agg["dominant_label"] == "positive"
    assert agg["positive_count"] == 6
    assert agg["neutral_count"] == 2
    assert agg["negative_count"] == 0
    assert agg["agreement_score"] == 6 / 8  # 0.75
    # teacher_soft_score = (6 - 0) / 8 = 0.75
    assert agg["teacher_soft_score"] == 0.75


def test_consistency_filtering():
    """Verify consistency filtering retains samples meeting the threshold."""
    df = pd.DataFrame([
        {"post_id": "p1", "dominant_count": 8, "dominant_label": "positive", "agreement_score": 1.0},
        {"post_id": "p2", "dominant_count": 6, "dominant_label": "negative", "agreement_score": 0.75},
        {"post_id": "p3", "dominant_count": 4, "dominant_label": "neutral", "agreement_score": 0.50},
    ])
    filtered, stats = filter_by_consistency(df, threshold=5, total_paths=8)
    assert len(filtered) == 2
    assert "p3" not in filtered["post_id"].values
    assert stats["retention_count"] == 2


if __name__ == "__main__":
    test_prompt_construction()
    test_demonstration_shuffle()
    test_strict_llm_parser_valid_json()
    test_strict_llm_parser_markdown_fence()
    test_strict_llm_parser_invalid_label_rejection()
    test_mock_llm_provider()
    test_multi_path_generation_and_schema()
    test_majority_voting_and_soft_score()
    test_consistency_filtering()
    print("All LLM pipeline unit tests passed successfully!")
