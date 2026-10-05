# System Architecture: Reddit Market Sentiment & Knowledge Distillation

**Reference:** Deng et al. (WWW '23 Companion, arXiv:2212.11311)  
**System:** Antigravity Financial Sentiment Analysis Platform

---

## 1. High-Level Architectural Flowchart

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. DATA INGESTION & PIPELINE                    │
│  Reddit Posts (WSB, stocks, investing) ──→ HTML/Emoji Normalization    │
│  CashTag/Ticker Extraction ──→ Stratified/Chronological Splitting      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. TEACHER LLM WEAK LABELING                    │
│  6-Shot Prompt (configs/prompts/sentiment_prompt.yaml)                 │
│  Provider Abstraction (OpenAI, Gemini, Claude, Ollama, Mock)           │
│  Stochastic Multi-Path Sampling (K=8 paths, Temperature=0.5)           │
│  Per-Path Tracking: post_id, path_id, rationale, label, confidence     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  3. AGGREGATION & SOFT SCORE GENERATION                │
│  Majority / Plurality Voting (Direct LLM Evaluation)                   │
│  Deterministic Tie-Breaking Strategy (neutral_default)                 │
│  Continuous Agreement Target: s = (pos - neg) / K ∈ [-1.0, 1.0]        │
│  Output: weak_labels.csv (positive_count, neutral_count, neg_count)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       4. CONSISTENCY FILTERING                         │
│  Filter Threshold: Dominant Class Count ≥ M (Default M = 5 out of 8)   │
│  Retention Tracking & Class Balance Reporting                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 5. STUDENT MODEL KNOWLEDGE DISTILLATION                │
│  Backbone Modernization: DistilBERT-base-uncased / DeBERTa-v3-small    │
│  Objective R1: Cross-Entropy Loss (Categorical Baseline)               │
│  Objective R2: Mean Squared Error Loss (MSE Regression Distillation)   │
│  Validation Threshold Search: Tuning θ ∈ [0.10, 0.40] for Macro F1     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│       6. BENCHMARK & EVALUATION      │  │    7. FINANCIAL MARKET ANALYSIS      │
│  Baselines: VADER, TF-IDF + LogReg/  │  │  Daily Sentiment Signals             │
│  SVM/NB, FinBERT-Prosus, FinBERT-HKUST│  │  Net Sentiment = Pos_Ratio - Neg_Ratio│
│  Teacher vs Student Comparisons      │  │  Forward Returns (1d, 3d, 5d)        │
│  Linguistic Error Categorization     │  │  Correlation, Lag & Event Studies    │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   8. INTERACTIVE STREAMLIT RESEARCH APP                │
│  8 Academic Pages: Overview, Live Predictor, Model Leaderboard,        │
│  Trends, Financial Returns, Error Diagnostics, Ablations, Methodology  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities & Component Layout

### 2.1 `src/data/`
- `ingestion.py`: Ingests raw parquet/json Reddit post archives, validates mandatory columns (`post_id`, `text`, `title`, `timestamp`, `subreddit`), and generates sample data for testing.
- `preprocessing.py`: Cleans raw social media text, converts financial emojis (🚀, 💎🙌, 📉) to semantic tokens, strips URLs, and removes deleted posts.
- `ticker_extraction.py`: Multi-stage ticker extraction prioritizing cashtags (`$NVDA`), resolving company names (e.g. "Apple" $\to$ `AAPL`), and filtering stop tickers.
- `splitting.py`: Implements leakage-free chronological splitting (for financial backtests) and stratified splitting (for sentiment classification balance).
- `validation.py`: Asserts schema completeness, timestamp monotonic ordering, and class distribution integrity.

### 2.2 `src/llm/`
- `client.py`: Abstract provider pattern (`BaseLLMProvider`) with unified implementations for `OpenAIProvider`, `GeminiProvider`, `AnthropicProvider`, `LocalProvider`, and `MockProvider`.
- `prompts.py`: Manages YAML prompt configuration (`configs/prompts/sentiment_prompt.yaml`), renders the 6 demonstrations (2 positive, 2 neutral, 2 negative), and supports demonstration shuffling for ablation studies.
- `parser.py`: Strict schema output validation enforcing `{"reasoning_summary": ..., "sentiment": "positive|neutral|negative"}`.
- `labeling.py`: Generates $K=8$ independent reasoning paths at $T=0.5$ with disk caching and retry logic.
- `aggregation.py`: Computes majority vote, handles ties explicitly, calculates soft agreement score, and generates `weak_labels.csv`.

### 2.3 `src/distillation/`
- `dataset.py`: PyTorch `TextRegressionDataset` and `TextClassificationDataset`, with consistency filtering ($\ge M/8$).
- `student_model.py`: Compact transformer encoder with regression or classification head.
- `train.py`: Training engine supporting MSE loss on continuous teacher targets and CrossEntropy on categorical targets.
- `evaluate.py`: Validation threshold search for $\theta$ and holdout test set evaluation.

### 2.4 `src/baselines/`
- `tfidf_lr.py`: TF-IDF n-gram vectorizer paired with L2-regularized Logistic Regression.
- `tfidf_svm.py`: TF-IDF paired with Linear Support Vector Classifier and probability calibration.
- `tfidf_nb.py`: TF-IDF paired with Multinomial Naive Bayes.
- `vader.py`: VADER lexicon baseline adapted with social financial sentiment terms.
- `finbert.py`: Pre-trained financial BERT transformers (`ProsusAI/finbert` and `yiyanghkust/finbert-tone`).

### 2.5 `src/experiments/`
- `ablation.py`: Executes reasoning paths ablation ($K \in \{1, 3, 5, 8, 16\}$), demonstration ordering, and filtering thresholds ($M \in \{5, 6, 7, 8\}$).
- `benchmark.py`: Evaluates all baselines, teacher variants, and student models on the same holdout test set.
- `error_analysis.py`: Categorizes model failure modes into sarcasm/memes, jargon, contradictory arguments, long DD posts, and ticker ambiguity.

### 2.6 `src/finance/`
- `market_data.py`: Downloads historical OHLCV data via yfinance and aligns post timestamps to NYSE/NASDAQ trading hours.
- `aggregation.py`: Computes daily ticker-level financial signals: `positive_ratio`, `negative_ratio`, `neutral_ratio`, `net_sentiment`, post volume, and rolling 1d/3d/7d sentiment.
- `returns.py`: Computes forward returns across 1d, 3d, and 5d trading day horizons.
- `analysis.py`: Computes Pearson/Spearman correlations, lead/lag relationships, OLS regressions, predictive comparisons, and event analyses.

---

## 3. Technical Constraints & Modernizations

1. **Teacher Modernization:** The paper's PaLM-540B is retired and inaccessible. We modernize with an open API architecture supporting state-of-the-art reasoning models (`gpt-4o-mini`, `gemini-1.5-flash`, `claude-3-5-sonnet`) while preserving the 6-shot CoT and 8-path sampling setup.
2. **Student Modernization:** Charformer (character-level T5, 102M) is not available or maintained in Hugging Face. We modernize with `distilbert-base-uncased` (66M) and `microsoft/deberta-v3-small` (44M), which offer superior sub-word tokenization and edge latency while preserving the core regression distillation hypothesis.
3. **Zero-Cost Offline Execution:** A deterministic `MockProvider` enables continuous testing and verification of the full pipeline without paid API access.
