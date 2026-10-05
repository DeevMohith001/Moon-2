"""
TF-IDF + Logistic Regression Baseline.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import compute_sentiment_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TFIDFLogisticRegressionBaseline:
    """TF-IDF + Logistic Regression Baseline."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
        ])
        self.is_fitted = False

    def fit(self, texts: list, labels: list):
        logger.info(f"Training TF-IDF + Logistic Regression on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: list) -> list:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict().")
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba().")
        return self.pipeline.predict_proba(texts)

    def evaluate(self, test_df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "sentiment_label") -> Dict[str, Any]:
        col = text_col if text_col in test_df.columns else "text"
        preds = self.predict(test_df[col].tolist())
        metrics = compute_sentiment_metrics(test_df[label_col].tolist(), preds)
        logger.info(f"TF-IDF + LogReg Test Accuracy: {metrics['accuracy']*100:.2f}%, Macro F1: {metrics['macro_f1']:.4f}")
        return metrics
