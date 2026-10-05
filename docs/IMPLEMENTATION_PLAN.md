# Implementation Plan: Reddit Financial Sentiment Analysis & Knowledge Distillation

**Project:** "LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models"  
**Academic Venue Reference:** WWW '23 Companion (arXiv:2212.11311)  
**Author / Team:** ML Research & Engineering Team  
**Date:** September 2026

---

## 1. Architectural Vision & Scope

The objective of this project is to build an end-to-end, production-grade, and academically rigorous financial sentiment analysis system for social media (Reddit). The project is divided into two distinct components:

1. **[PAPER REPRODUCTION]:** Direct implementation of the methodology from Deng et al. (WWW '23 Companion), including:
   - Configurable LLM teacher with few-shot in-context learning
   - Chain-of-Thought (CoT) / TL;DR financial reasoning
   - Stochastic multi-path sampling ($K=8$ paths at temperature $T=0.5$)
   - Direct LLM evaluation via majority voting
   - Continuous soft agreement score calculation
   - Weak label dataset generation (`weak_labels.csv`)
   - Consistency filtering thresholds ($M \in \{5, 6, 7, 8\}$ out of 8)
   - Student model distillation using regression loss (MSE) on soft targets vs. classification baseline
   - Evaluation against FinBERT and external FiQA benchmarks.

2. **[PROJECT EXTENSION]:** Advanced capabilities not explored in the original paper:
   - Traditional ML baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes) and lexicon baseline (VADER)
   - Modern FinBERT baselines (ProsusAI and HKUST)
   - Empirical financial market analysis: daily sentiment aggregation, forward returns (1d, 3d, 5d), correlation (Pearson/Spearman), lag analysis, event studies, rolling sentiment windows (1d, 3d, 7d), and predictive regression
   - Comprehensive error and disagreement diagnostics across linguistic and financial categories
   - Interactive 8-page Streamlit research dashboard
   - Complete experiment tracking, caching, offline mock fallback, and Docker containerization.

---

## 2. Directory Layout & Module Organization

To satisfy both modular architectural guidelines and ensure backwards compatibility, the workspace is structured as follows:

```
project_root/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── CITATIONS.md
│
├── configs/
│   ├── data.yaml                    # Data ingestion, cleaning & split parameters
│   ├── model.yaml                   # Student model backbones & hyperparameters
│   ├── training.yaml                # Distillation training & optimization settings
│   ├── experiments.yaml             # Ablation & benchmark matrices
│   └── prompts/
│       └── sentiment_prompt.yaml    # 6-shot demonstrations, CoT instructions
│
├── data/
│   ├── raw/                         # Raw Reddit datasets
│   ├── interim/                     # Cleaned posts with tickers
│   ├── processed/                   # Train/Val/Test splits & merged return datasets
│   └── external/                    # FiQA benchmark datasets
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis & vocabulary
│   ├── 02_llm_labeling.ipynb        # 8-path LLM teacher weak labeling
│   ├── 03_student_training.ipynb    # Student regression distillation
│   ├── 04_evaluation.ipynb          # Benchmark evaluation & error analysis
│   └── 05_financial_analysis.ipynb  # Sentiment vs. stock return dynamics
│
├── src/
│   ├── data/
│   │   ├── ingestion.py             # Data loading & standardization
│   │   ├── preprocessing.py         # Text cleaning, emoji normalization
│   │   ├── ticker_extraction.py     # High-precision ticker detection
│   │   ├── splitting.py             # Chronological & stratified splitters
│   │   └── validation.py            # Dataset integrity assertions
│   │
│   ├── llm/
│   │   ├── client.py                # BaseLLMProvider, OpenAI, Gemini, Claude, Mock
│   │   ├── prompts.py               # Prompt formatting & demonstration loading
│   │   ├── parser.py                # Strict JSON / schema output parser
│   │   ├── labeling.py              # 8-path reasoning generation with cache
│   │   └── aggregation.py           # Majority voting & soft score computation
│   │
│   ├── distillation/
│   │   ├── dataset.py               # PyTorch regression/classification datasets
│   │   ├── student_model.py         # Student encoder with regression/class head
│   │   ├── train.py                 # Training loops for regression & classification
│   │   └── evaluate.py              # Threshold grid search & test evaluation
│   │
│   ├── baselines/
│   │   ├── tfidf_lr.py              # TF-IDF + Logistic Regression
│   │   ├── tfidf_svm.py             # TF-IDF + Linear SVM
│   │   ├── tfidf_nb.py              # TF-IDF + Naive Bayes
│   │   ├── vader.py                 # Financial-adapted VADER
│   │   └── finbert.py               # FinBERT-ProsusAI & FinBERT-HKUST
│   │
│   ├── experiments/
│   │   ├── ablation.py              # Paths ablation, demo ordering, filtering
│   │   ├── benchmark.py             # Master evaluation across all models
│   │   └── error_analysis.py        # Categorized linguistic error diagnosis
│   │
│   ├── finance/
│   │   ├── market_data.py           # Historical prices & trading calendar alignment
│   │   ├── aggregation.py           # Daily sentiment scores & rolling metrics
│   │   ├── returns.py               # Forward returns (1d, 3d, 5d) & volatility
│   │   └── analysis.py              # Correlation, regression & event studies
│   │
│   └── utils/
│       ├── logging.py               # Standardized logging configuration
│       ├── seed.py                  # Seed pinning across all frameworks
│       └── config.py                # YAML configuration loader
│
├── dashboard/
│   └── app.py                       # 8-Page interactive Streamlit research app
│
├── outputs/                         # Experiment outputs, metrics, and models
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── reports/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_llm_parser.py
│   ├── test_aggregation.py
│   ├── test_distillation.py
│   ├── test_baselines.py
│   ├── test_finance.py
│   └── test_smoke_pipeline.py       # End-to-end zero-cost smoke test
│
└── docs/
    ├── PAPER_METHODOLOGY.md
    ├── IMPLEMENTATION_PLAN.md
    ├── ARCHITECTURE.md
    ├── EXPERIMENTS.md
    ├── DATASET.md
    └── LIMITATIONS.md
```

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Foundation & Data Pipeline
- Implement `configs/data.yaml` and standardize ingestion in `src/data/ingestion.py`.
- Implement robust text normalization, financial emoji preservation, and regex ticker extraction in `src/data/preprocessing.py` and `src/data/ticker_extraction.py`.
- Ensure clean splitting (chronological and stratified) with zero data leakage in `src/data/splitting.py`.

### Phase 2: Teacher LLM Abstraction & Prompting
- Implement `src/llm/client.py` featuring `BaseLLMProvider` with:
  - `OpenAIProvider`
  - `GeminiProvider` (Google GenAI)
  - `AnthropicProvider`
  - `LocalProvider` (Ollama)
  - `MockLLMProvider` (deterministic, zero-cost, high-fidelity offline mode).
- Create `configs/prompts/sentiment_prompt.yaml` with the paper's exact 6 demonstration structure (2 positive, 2 neutral, 2 negative) with CoT reasoning.
- Implement `src/llm/parser.py` with strict schema validation enforcing normalized labels `positive`, `neutral`, and `negative`.

### Phase 3: Multi-Path Reasoning, Majority Voting & Soft Scores
- Implement `src/llm/labeling.py` for $K=8$ path generation with temperature $T=0.5$.
- Implement disk caching (SHA-256 hash of input + model + hyperparams), exponential backoff, and progress tracking.
- Implement `src/llm/aggregation.py` with:
  - Plurality / majority voting with documented tie-breaking.
  - Calculation of `positive_count`, `neutral_count`, `negative_count`, `agreement_score`, and continuous `teacher_soft_score` in $[-1.0, 1.0]$.
  - Generation of `weak_labels.csv`.

### Phase 4: Consistency Filtering & Distillation Pipeline
- Implement consistency filtering in `src/distillation/dataset.py` supporting thresholds $M \in \{5, 6, 7, 8\}$ out of 8.
- Implement `src/distillation/student_model.py` with configurable modern backbones (DistilBERT-base-uncased, DeBERTa-v3-small, MiniLM) and dual heads:
  - Regression head (outputting continuous scalar $\hat{s} \in \mathbb{R}$)
  - Classification head (outputting 3-class logits).
- Implement training loop in `src/distillation/train.py`:
  - **Objective R1:** CrossEntropyLoss on categorical teacher labels.
  - **Objective R2:** MSELoss on continuous soft teacher targets.
- Implement threshold search in `src/distillation/evaluate.py` to tune class boundaries on validation set.

### Phase 5: Baselines Implementation
- Traditional ML in `src/baselines/`:
  - `tfidf_lr.py`: Logistic Regression
  - `tfidf_svm.py`: Linear SVM
  - `tfidf_nb.py`: Multinomial Naive Bayes.
- Lexicon baseline in `src/baselines/vader.py` with financial terminology mapping.
- Financial transformer baselines in `src/baselines/finbert.py`:
  - `ProsusAI/finbert`
  - `yiyanghkust/finbert-tone`.
- Unified evaluation interface producing standardized metrics: Accuracy, Precision, Recall, Macro F1, Weighted F1, Per-class F1, Confusion Matrix.

### Phase 6: Paper-Style Experiments & Ablation Studies
- Implement `src/experiments/ablation.py`:
  - Exp 1: Few-shot baseline
  - Exp 2: Few-shot + CoT reasoning
  - Exp 3: Few-shot + CoT + multiple reasoning paths
  - Exp 4: Reasoning paths ablation ($K \in \{1, 3, 5, 8, 16\}$)
  - Exp 5: Demonstration order permutation
  - Exp 6: Consistency filtering thresholds ($M \in \{5, 6, 7, 8\}$)
  - Exp 7: Classification vs. Regression distillation
  - Exp 8: Student backbone comparisons.
- Implement `src/experiments/error_analysis.py` categorizing errors: sarcasm, memes, long due diligence posts, mixed arguments, options jargon, and ticker ambiguity.

### Phase 7: Financial Market Analysis Extension
- Implement `src/finance/`:
  - `market_data.py`: Stock price acquisition via yfinance with NYSE/NASDAQ trading day alignment.
  - `aggregation.py`: Daily sentiment scores, net sentiment ($\text{pos} - \text{neg}$), rolling sentiment (1d, 3d, 7d).
  - `returns.py`: 1-day, 3-day, 5-day forward returns and annualized volatility.
  - `analysis.py`: Correlation (Pearson and Spearman), OLS regression, predictive modeling, and high-attention event studies.
  - Prominent academic disclaimers (not financial advice).

### Phase 8: Interactive Streamlit Dashboard
- Refactor and extend `dashboard/app.py` into an 8-page academic application:
  - **Page 1: Overview** (Project architecture, datasets, summary statistics)
  - **Page 2: Sentiment Analysis** (Interactive single-post predictor with confidence and model choice)
  - **Page 3: Model Comparison** (Comprehensive leaderboard across accuracy, macro F1, latency)
  - **Page 4: Sentiment Trends** (Ticker-specific time series, post velocity, sentiment distributions)
  - **Page 5: Financial Analysis** (Sentiment vs. forward returns, rolling sentiment, event studies)
  - **Page 6: Error Analysis** (Failure mode taxonomy, model disagreement inspector)
  - **Page 7: Research Experiments** (Visualizations of ablations: paths, filtering, loss functions)
  - **Page 8: About / Methodology** (Paper reproduction mapping, extensions, ethics & limitations).

### Phase 9: Testing, Verification & Containerization
- Unit tests covering preprocessing, parser, labeling, aggregation, distillation, baselines, and finance.
- End-to-end zero-cost smoke test (`tests/test_smoke_pipeline.py`) running a complete mini-pipeline on 10 samples via `MockLLMProvider`.
- Create `Dockerfile` and `docker-compose.yml`.
- Verify all requirements, documentation, and reproducibility.

---

## 4. Key Methodological Constants & Defaults

- **Teacher Sampling Temperature:** $0.5$
- **Number of Reasoning Paths:** $K = 8$
- **Default Consistency Threshold:** $M = 5$ out of $8$ ($\ge 62.5\%$ agreement)
- **Sentiment Target Range:** $[-1.0, 1.0]$ where $-1.0 = \text{bearish}$, $0.0 = \text{neutral}$, $+1.0 = \text{bullish}$
- **Student Learning Rate:** $5 \times 10^{-5}$
- **Student Batch Size:** $16$ (CPU-compatible) / $64$ (GPU-capable)
- **Random Seed:** $42$ across Python, NumPy, PyTorch, and scikit-learn.
