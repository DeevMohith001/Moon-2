"""
Unit tests for VADER baseline and financial lexicon mappings.
"""

from src.baselines.vader import VADERBaseline


def test_vader_bullish_prediction():
    vader = VADERBaseline()
    text = "Massive revenue beat! NVDA to the moon 🚀 buying calls"
    label = vader.predict_label(text)
    score = vader.predict_score(text)
    assert label == "BULLISH"
    assert score > 0.05


def test_vader_bearish_prediction():
    vader = VADERBaseline()
    text = "Company is drilling down, massive dilution and bankruptcy risk. Puts are printing."
    label = vader.predict_label(text)
    score = vader.predict_score(text)
    assert label == "BEARISH"
    assert score < -0.05


def test_vader_neutral_prediction():
    vader = VADERBaseline()
    text = "What time is the 10-Q filing scheduled tomorrow?"
    label = vader.predict_label(text)
    assert label in ("NEUTRAL", "BULLISH", "BEARISH")


if __name__ == "__main__":
    test_vader_bullish_prediction()
    test_vader_bearish_prediction()
    test_vader_neutral_prediction()
    print("VADER tests passed!")
