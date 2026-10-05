"""
Unit tests for evaluation metrics.
"""

from src.evaluation.metrics import compute_sentiment_metrics


def test_metrics_computation():
    y_true = ["BULLISH", "BEARISH", "NEUTRAL", "BULLISH"]
    y_pred = ["BULLISH", "NEUTRAL", "NEUTRAL", "BULLISH"]
    m = compute_sentiment_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert "macro_f1" in m
    assert "per_class" in m
    assert "BULLISH" in m["per_class"]
    assert m["per_class"]["BULLISH"]["recall"] == 1.0


if __name__ == "__main__":
    test_metrics_computation()
    print("Metrics tests passed!")
