"""Baseline Sentiment Models for Social Media Financial Text.

Submodules are intentionally imported lazily so optional dependencies for one
baseline do not block importing another.
"""

__all__ = [
    "TFIDFLogisticRegressionBaseline",
    "TFIDFLinearSVMBaseline",
    "TFIDFNaiveBayesBaseline",
    "VADERBaseline",
    "FinBERTBaseline",
]
