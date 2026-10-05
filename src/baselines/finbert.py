"""FinBERT baselines: ProsusAI/finbert and yiyanghkust/finbert-tone."""

from typing import Iterable, List, Tuple


class FinBERTBaseline:
    """Lazy Transformers wrapper that normalizes FinBERT outputs to project labels."""

    LABEL_MAP = {
        "positive": "BULLISH",
        "neutral": "NEUTRAL",
        "negative": "BEARISH",
        "bullish": "BULLISH",
        "bearish": "BEARISH",
        "label_0": "BEARISH",
        "label_1": "NEUTRAL",
        "label_2": "BULLISH",
    }

    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,
                device=self.device,
                truncation=True,
            )
        return self._pipeline

    def _normalize_label(self, label: str) -> str:
        return self.LABEL_MAP.get(str(label).strip().lower(), str(label).upper())

    def predict_batch(self, texts: Iterable[str]) -> Tuple[List[str], List[float], List[dict]]:
        outputs = self.pipeline([str(text) for text in texts])
        labels = []
        confidences = []
        probabilities = []

        for output in outputs:
            scores = output if isinstance(output, list) else [output]
            probs = {
                self._normalize_label(item["label"]): float(item["score"])
                for item in scores
            }
            if probs:
                label, confidence = max(probs.items(), key=lambda item: item[1])
            else:
                label, confidence = "NEUTRAL", 0.0
            labels.append(label)
            confidences.append(confidence)
            probabilities.append(probs)

        return labels, confidences, probabilities

    def predict(self, texts: Iterable[str]) -> List[str]:
        labels, _, _ = self.predict_batch(texts)
        return labels

    def predict_label(self, text: str) -> str:
        return self.predict([text])[0]

__all__ = ["FinBERTBaseline"]
