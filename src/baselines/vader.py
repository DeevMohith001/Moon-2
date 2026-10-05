"""VADER lexicon baseline with finance-specific cue adjustments."""

from typing import Iterable, List, Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


class VADERBaseline:
    """Small wrapper around VADER that returns project label names."""

    BULLISH_TERMS = {
        "beat": 1.4,
        "beats": 1.4,
        "breakout": 1.8,
        "bullish": 2.2,
        "calls": 1.5,
        "long": 1.0,
        "moon": 2.0,
        "rocket": 1.8,
        "undervalued": 1.4,
        "upgrade": 1.4,
        "upgraded": 1.4,
        "🚀": 2.2,
    }
    BEARISH_TERMS = {
        "bankrupt": -2.4,
        "bankruptcy": -2.4,
        "bearish": -2.2,
        "dilution": -1.8,
        "downgrade": -1.5,
        "downgraded": -1.5,
        "drilling": -1.7,
        "dump": -1.5,
        "miss": -1.2,
        "puts": -1.5,
        "short": -1.0,
        "tank": -1.8,
    }

    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05):
        self.analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def _financial_adjustment(self, text: str) -> float:
        tokens = text.lower().split()
        score = 0.0
        for token in tokens:
            cleaned = token.strip(".,!?;:()[]{}\"'")
            score += self.BULLISH_TERMS.get(cleaned, 0.0)
            score += self.BEARISH_TERMS.get(cleaned, 0.0)
        if "🚀" in text:
            score += self.BULLISH_TERMS["🚀"]
        return max(min(score / 10.0, 0.5), -0.5)

    def predict_score(self, text: str) -> float:
        vader_score = (
            self.analyzer.polarity_scores(str(text))["compound"]
            if self.analyzer
            else 0.0
        )
        score = vader_score + self._financial_adjustment(str(text))
        return max(min(score, 1.0), -1.0)

    def predict_label(self, text: str) -> str:
        score = self.predict_score(text)
        if score > self.positive_threshold:
            return "BULLISH"
        if score < self.negative_threshold:
            return "BEARISH"
        return "NEUTRAL"

    def predict(self, texts: Iterable[str]) -> List[str]:
        return [self.predict_label(text) for text in texts]

    def predict_batch(self, texts: Iterable[str]) -> Tuple[List[str], List[float]]:
        labels = []
        scores = []
        for text in texts:
            score = self.predict_score(text)
            scores.append(score)
            labels.append(self.predict_label(text))
        return labels, scores

__all__ = ["VADERBaseline"]
