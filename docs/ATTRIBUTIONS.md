# Attributions and References

## 1. Academic Research Papers
- **Deng, X., Bashivan, P., & Precup, D. (2023).** *What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment.* arXiv:2212.11311.
  - Core methodology reproduced: 8-reasoning path teacher weak labeling, consensus threshold filtering (>= 5/8 agreement), and student model distillation using regression loss on continuous agreement ratios.
- **Araci, D. (2019).** *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
  - Used for `ProsusAI/finbert` baseline model.
- **Yang, Y., Uy, M. C. S., & Huang, A. (2020).** *FinBERT: A Large Language Model for Extracting Information from Financial Text.* Contemporary Accounting Research.
  - Used for `yiyanghkust/finbert-tone` baseline model.
- **Hutto, C., & Gilbert, E. (2014).** *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* Eighth International AAAI Conference on Weblogs and Social Media (ICWSM-14).
  - Used for VADER lexicon-based sentiment baseline.

---

## 2. Open-Source Libraries and Tooling
- **Transformers & Hugging Face Hub:** Sequence classification pipelines, tokenizers, and model checkpoint hosting.
- **PyTorch:** Model distillation and regression loss training.
- **scikit-learn:** TF-IDF feature extraction, Logistic Regression, Linear SVM, and multi-class classification metrics.
- **yfinance:** Acquisition of historical equity OHLCV prices and adjusted returns.
- **Streamlit & Plotly:** Interactive dark-mode research intelligence dashboard and econometric charting.
- **vaderSentiment:** Lexicon polarity scoring engine.
- **Pandas & NumPy:** Parquet dataset engineering, temporal trading calendar alignment, and time-series aggregation.
