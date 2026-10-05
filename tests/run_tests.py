"""
Direct test runner to execute all unit tests and end-to-end smoke tests.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_preprocessing import (
    test_clean_text_html_and_urls,
    test_clean_text_emoji_preservation,
    test_extract_cashtags,
    test_detect_tickers_cashtag_priority,
    test_detect_tickers_company_name,
    test_stop_tickers_filtered,
    test_create_splits_ratio,
)
from tests.test_vader import (
    test_vader_bullish_prediction,
    test_vader_bearish_prediction,
    test_vader_neutral_prediction,
)
from tests.test_finance import (
    test_trading_day_alignment,
    test_sentiment_aggregation_math,
)
from tests.test_metrics import (
    test_metrics_computation,
)
from tests.test_llm_pipeline import (
    test_prompt_construction,
    test_demonstration_shuffle,
    test_strict_llm_parser_valid_json,
    test_strict_llm_parser_markdown_fence,
    test_strict_llm_parser_invalid_label_rejection,
    test_mock_llm_provider,
    test_multi_path_generation_and_schema,
    test_majority_voting_and_soft_score,
    test_consistency_filtering,
)
from tests.test_smoke_pipeline import (
    test_end_to_end_smoke_pipeline,
)

from tests.test_data_integrity import (
    test_assert_real_data_rejects_synthetic,
    test_assert_real_data_accepts_real,
    test_create_splits_zero_leakage_and_mutually_exclusive,
    test_fiqa_loader_protocol_and_subtasks,
    test_standardize_hf_reddit_schema,
)

test_funcs = [
    test_assert_real_data_rejects_synthetic,
    test_assert_real_data_accepts_real,
    test_create_splits_zero_leakage_and_mutually_exclusive,
    test_fiqa_loader_protocol_and_subtasks,
    test_standardize_hf_reddit_schema,
    test_clean_text_html_and_urls,
    test_clean_text_emoji_preservation,
    test_extract_cashtags,
    test_detect_tickers_cashtag_priority,
    test_detect_tickers_company_name,
    test_stop_tickers_filtered,
    test_create_splits_ratio,
    test_vader_bullish_prediction,
    test_vader_bearish_prediction,
    test_vader_neutral_prediction,
    test_trading_day_alignment,
    test_sentiment_aggregation_math,
    test_metrics_computation,
    test_prompt_construction,
    test_demonstration_shuffle,
    test_strict_llm_parser_valid_json,
    test_strict_llm_parser_markdown_fence,
    test_strict_llm_parser_invalid_label_rejection,
    test_mock_llm_provider,
    test_multi_path_generation_and_schema,
    test_majority_voting_and_soft_score,
    test_consistency_filtering,
    test_end_to_end_smoke_pipeline,
]

failed = 0
passed = 0
for fn in test_funcs:
    try:
        fn()
        print(f"PASSED: {fn.__name__}")
        passed += 1
    except Exception as e:
        print(f"FAILED: {fn.__name__} - {e}")
        failed += 1

print(f"\n==========================================")
print(f"Test Run Summary: {passed} passed, {failed} failed.")
print(f"==========================================\n")
if failed > 0:
    sys.exit(1)
