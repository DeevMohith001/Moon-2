# PROJECT PLAN
## LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models
### LLM for Financial Applications — IIIT Dharwad

---

## Executive Summary

This project reproduces and extends the methodology of Deng et al. (2023), "What do LLMs
Know about Financial Markets?" (arXiv:2212.11311). The original paper uses PaLM-540B to
generate weak financial sentiment labels for Reddit posts, then distills into a smaller
Charformer student model using regression loss. We adapt this methodology using publicly
available LLMs and datasets, and extend with financial market analysis that the original
paper did not investigate.

---

## Environment Assessment

| Component | Status | Detail |
|-----------|--------|--------|
| Python | ✅ | 3.14.3 |
| PyTorch | ✅ | 2.11.0 (CPU only) |
| Transformers | ✅ | 5.4.0 |
| scikit-learn | ✅ | 1.8.0 |
| Pandas/NumPy | ✅ | 2.3.3 / 2.4.3 |
| Streamlit | ✅ | 1.55.0 |
| OpenAI SDK | ✅ | 2.30.0 |
| Anthropic SDK | ✅ | 0.88.0 |
| NLTK | ✅ | 3.9.4 |
| Matplotlib | ✅ | 3.10.8 |
| Plotly | ✅ | 6.6.0 |
| SciPy | ✅ | 1.17.1 |
| GPU | ❌ | CPU only — constrains FinBERT/distillation speed |
| VADER | ❌ | Needs install |
| yfinance | ❌ | Needs install |
| seaborn | ❌ | Needs install |
| pytest | ❌ | Needs install |
| xgboost | ❌ | Needs install (optional) |
| OpenAI API Key | ❓ | Not checked — needed for LLM experiments |
| Anthropic API Key | ❓ | Not checked — needed for LLM experiments |

## Gap Analysis: Paper vs. Our Reproduction

| Paper Component | Available? | Our Approach |
|----------------|-----------|--------------|
| PaLM-540B | ❌ | Use GPT-4/Claude/open-source via API |
| Charformer (social-media pre-trained) | ❌ | DistilBERT or RoBERTa-base |
| Internal Reddit dataset (20K train) | ❌ | Public Reddit dataset from Kaggle/HuggingFace |
| Internal topic classifier | ❌ | Subreddit-based + ticker detection |
| Internal stock popularity filter | ❌ | Top-traded tickers (S&P 500 subset) |
| 100-post expert-annotated test set | ❌ | Create our own (target 500+) |
| CoT + 6-shot prompt design | ✅ | Reproduce faithfully |
| 8 reasoning paths + majority vote | ✅ | Reproduce faithfully |
| Regression loss distillation | ✅ | Reproduce faithfully |
| FinBERT baselines | ✅ | ProsusAI/finbert + yiyanghkust/finbert-tone |
| FiQA benchmark | ✅ | Publicly available |

---

## Phase Plan

### PHASE 0: Paper Analysis + Research Design
**Status:** ✅ COMPLETE

**Deliverables:**
- [x] Paper read and analyzed
- [x] `docs/PAPER_REVIEW.md` created
- [x] `PROJECT_PLAN.md` created
- [x] Environment assessed
- [x] Gap analysis documented

---

### PHASE 1: Dataset Acquisition and Preprocessing
**Status:** ✅ COMPLETE

**Objective:** Acquire a public Reddit financial dataset and build robust preprocessing.

**Dataset Strategy:**
1. **Primary:** Search for existing labeled Reddit financial sentiment datasets on
   Kaggle/HuggingFace (e.g., Reddit WallStreetBets posts with timestamps)
2. **Secondary:** Use raw Reddit data dumps (pushshift archives) and filter for
   financial subreddits
3. **Supplementary:** FiQA benchmark for cross-dataset evaluation

**Target subreddits:** r/wallstreetbets, r/stocks, r/investing, r/StockMarket

**Target dataset size:**
- Training: 10,000-50,000 posts
- Validation: 1,000-5,000 posts
- Test: 500-2,000 posts (including human-annotated subset)

**Deliverables:**
- [x] Dataset downloaded and documented
- [x] `src/data/ingestion.py` — data loading and format standardization
- [x] `src/data/preprocessing.py` — cleaning pipeline
- [x] `src/data/ticker_detection.py` — ticker/company identification
- [x] `src/data/validation.py` — data quality checks
- [x] `data/raw/`, `data/interim/`, `data/processed/` populated
- [x] `docs/DATASET.md` — dataset documentation
- [x] Unit tests for preprocessing
- [x] Data quality report

**Preprocessing pipeline:**
```
Raw Reddit data
    -> Remove duplicates
    -> Remove empty/deleted posts
    -> Clean HTML/markdown
    -> Handle URLs (remove or replace)
    -> Handle emojis (preserve sentiment-bearing ones)
    -> Normalize whitespace
    -> Normalize timestamps to UTC
    -> Filter very short texts (<10 chars)
    -> Detect tickers
    -> Normalize ticker symbols
    -> Remove clearly non-financial posts (if subreddit is mixed)
    -> Output: processed DataFrame
```

---

### PHASE 2: Traditional NLP Baselines
**Status:** ✅ COMPLETE
**Depends on:** Phase 1

**Models:**
1. **VADER** — lexicon-based sentiment baseline
2. **TF-IDF + Logistic Regression** — classical ML
3. **TF-IDF + Linear SVM** — classical ML

**Deliverables:**
- [x] `src/models/vader_model.py`
- [x] `src/models/classical_ml.py`
- [x] Models saved to `models/`
- [x] Evaluation results in `results/metrics/`
- [x] Confusion matrices in `results/figures/`

**Notes:**
- VADER will use the NLTK/vaderSentiment library
- Map VADER compound scores to 3 classes: positive->Bullish, negative->Bearish, neutral->Neutral
- TF-IDF models trained on training set only
- Hyperparameter tuning on validation set only (grid search)
- No test set access during tuning

---

### PHASE 3: Financial Language Model Baseline
**Status:** ✅ COMPLETE
**Depends on:** Phase 1

**Models:**
1. **ProsusAI/finbert** — FinBERT (Araci, 2019)
2. **yiyanghkust/finbert-tone** — FinBERT-HKUST (Yang et al., 2020)

**Deliverables:**
- [x] `src/models/finbert.py`
- [x] Inference pipeline (batch processing, CPU-optimized)
- [x] Evaluation results in `results/metrics/`
- [x] Comparison with Phase 2 baselines

**Notes:**
- CPU inference will be slow; implement batching and progress reporting
- Use configurable batch size (start small: 8-16)
- Cache predictions to avoid recomputation

---

### PHASE 4: General LLM Experiments
**Status:** ✅ COMPLETE
**Depends on:** Phase 1

**Experiments:**

| ID | Strategy | Description |
|----|----------|-------------|
| A | Zero-shot | Simple sentiment classification prompt |
| B | Few-shot | 6-shot (2 per class), matching paper's approach |
| C | Financial-context | Include market terminology definitions |
| D | Structured-output | JSON output with ticker, sentiment, confidence, evidence |
| E | Reasoning-assisted | CoT reasoning before classification (paper's approach) |

**LLM Provider Abstraction:**
```
src/models/llm/
    base.py          — abstract LLM interface
    openai_model.py  — OpenAI API (GPT-4, GPT-3.5)
    anthropic_model.py — Anthropic API (Claude)
    local_model.py   — Ollama/HuggingFace local models (fallback)
```

**Deliverables:**
- [x] `src/models/llm/` package
- [x] 5 versioned prompts in `prompts/`
- [x] LLM output validation + retry logic
- [x] Caching system
- [x] Rate limiting + exponential backoff
- [x] Cost estimation
- [x] `configs/llm_config.yaml`
- [x] `.env.example`
- [x] Results for each experiment (start with 100-post debug run)

---

### PHASE 5: Original Paper Reproduction
**Status:** ✅ COMPLETE
**Depends on:** Phases 1, 4

**Pipeline (reproducing paper methodology):**
```
Reddit posts
    -> LLM (with CoT, 6-shot, temperature=0.5)
    -> 8 reasoning paths per post
    -> Majority vote -> weak labels
    -> Filter (>=5/8 consistent)
    -> Soft scores (agreement ratio)
    -> Student model (regression loss)
    -> Sentiment classifier
```

**Our adaptations:**
- LLM: GPT-4 or Claude (instead of PaLM-540B)
- Student: DistilBERT (instead of Charformer)
- Dataset: public Reddit data (instead of proprietary)

**Deliverables:**
- [x] `src/models/distillation.py`
- [x] `experiments/reproduction/` with full pipeline
- [x] Documented deviations from paper
- [x] Student model trained and evaluated
- [x] Comparison: direct LLM vs distilled student

---

### PHASE 6: Extended Experiments
**Status:** ✅ COMPLETE
**Depends on:** Phases 2-5

**Experiments:**
1. LLM consistency test (multiple runs on same data)
2. Confidence calibration analysis
3. Multi-model comparison table
4. Cross-dataset evaluation (Reddit -> FiQA)

**Deliverables:**
- [x] `src/evaluation/calibration.py`
- [x] Consistency analysis results
- [x] Calibration plots (reliability diagrams)
- [x] ECE calculation where applicable
- [x] Comprehensive model comparison table

---

### PHASE 7: Financial Market Analysis
**Status:** ✅ COMPLETE
**Depends on:** Phases 1-4

**Pipeline:**
```
Daily sentiment scores (per ticker)
    + Historical stock prices (yfinance)
    -> Temporal alignment (market hours)
    -> Forward returns (1d, 3d, 5d)
    -> Correlation analysis
    -> Regression analysis
    -> Predictive experiments
    -> Event analysis
```

**Deliverables:**
- [x] `src/finance/prices.py` — stock price data acquisition
- [x] `src/finance/alignment.py` — timestamp to trading day alignment
- [x] `src/finance/aggregation.py` — daily sentiment aggregation
- [x] `src/finance/returns.py` — return calculations
- [x] `src/finance/statistics.py` — correlation, regression
- [x] `src/finance/event_analysis.py` — high-activity day analysis
- [x] Results in `results/`

**Statistical analyses:**
- Pearson and Spearman correlation (sentiment to return)
- Simple OLS regression: Return = B0 + B1(Sentiment) + e
- Predictive model comparison:
  - Model A: market features only
  - Model B: sentiment only
  - Model C: market + sentiment
- Event analysis: high-attention days

---

### PHASE 8: Evaluation and Statistical Analysis
**Status:** ✅ COMPLETE
**Depends on:** Phases 2-7

**Deliverables:**
- [x] `src/evaluation/metrics.py` — unified metrics computation
- [x] `src/evaluation/error_analysis.py` — categorized error analysis
- [x] `results/sentiment_metrics.csv`
- [x] `results/error_analysis.csv`
- [x] `results/experiment_results.csv`
- [x] Publication-quality figures
- [x] `docs/EXPERIMENTS.md`

**Error categories to analyze:**
- Sarcasm/irony
- Financial jargon
- Mixed sentiment
- Ambiguous ticker
- Jokes/memes
- Implicit sentiment
- Market-wide vs company-specific

---

### PHASE 9: Interactive Streamlit Dashboard
**Status:** ✅ COMPLETE
**Depends on:** Phases 1-8

**Sections:**
A. Overview — total posts, stocks, date range, sentiment distribution
B. Stock Selector — dropdown filters
C. Sentiment Trend — line chart over time
D. Sentiment Distribution — bullish/neutral/bearish breakdown
E. Reddit Activity — posts per day
F. Stock Price — price overlay
G. Sentiment vs Return — scatter plot with correlation
H. Model Comparison — metrics table
I. Post Explorer — inspect individual posts
J. Error Analysis — model disagreement examples

**Deliverables:**
- [x] `dashboard/app.py`
- [x] Clear disclaimer: "Not Financial Advice"
- [x] All visualizations working with actual data

---

### PHASE 10: Testing, Documentation and Reproducibility
**Status:** ✅ COMPLETE
**Depends on:** All phases

**Deliverables:**
- [x] `tests/` with comprehensive unit tests
- [x] `pytest` passing
- [x] `requirements.txt` (pinned versions)
- [x] `environment.yml` (optional)
- [x] `.env.example`
- [x] `.gitignore`
- [x] `README.md` with full setup/run instructions
- [x] `docs/METHODOLOGY.md`
- [x] `docs/ANNOTATION_GUIDELINES.md`
- [x] `docs/LIMITATIONS.md`
- [x] `docs/ATTRIBUTIONS.md`
- [x] All experiment configurations saved

---

## Experimental Matrix

### Sentiment Classification Experiments

| Exp ID | Model | Type | Prompt | Training Data | Eval Set |
|--------|-------|------|--------|---------------|----------|
| B1 | VADER | Lexicon | N/A | N/A | Test |
| B2 | TF-IDF + LogReg | ML | N/A | Train | Test |
| B3 | TF-IDF + SVM | ML | N/A | Train | Test |
| B4 | FinBERT-ProsusAI | Transformer | N/A | Pre-trained | Test |
| B5 | FinBERT-HKUST | Transformer | N/A | Pre-trained | Test |
| L1 | LLM (zero-shot) | LLM | v1 | N/A | Test |
| L2 | LLM (few-shot) | LLM | v1 | N/A | Test |
| L3 | LLM (financial-context) | LLM | v1 | N/A | Test |
| L4 | LLM (structured-output) | LLM | v1 | N/A | Test |
| L5 | LLM (CoT reasoning) | LLM | v1 | N/A | Test |
| R1 | Student (distilled, classification loss) | Transformer | N/A | LLM-labeled | Test |
| R2 | Student (distilled, regression loss) | Transformer | N/A | LLM-labeled | Test |

### Financial Analysis Experiments

| Exp ID | Analysis | Tickers | Horizon | Method |
|--------|----------|---------|---------|--------|
| F1 | Correlation | Top-5 | 1d, 3d, 5d | Pearson, Spearman |
| F2 | Regression | Top-5 | 1d | OLS |
| F3 | Prediction (market only) | Top-5 | 1d | Logistic Regression |
| F4 | Prediction (sentiment only) | Top-5 | 1d | Logistic Regression |
| F5 | Prediction (combined) | Top-5 | 1d | Logistic Regression |
| F6 | Event analysis | All | 1d, 3d | Percentile-based |
| F7 | Engagement weighting | Top-5 | 1d | Weighted vs unweighted |

---

## Research Questions Mapping

| RQ | Question | Phase(s) |
|----|----------|----------|
| RQ1 | How accurately can different approaches classify Reddit sentiment? | 2, 3, 4, 5, 8 |
| RQ2 | FinBERT vs general LLMs? | 3, 4, 8 |
| RQ3 | Zero-shot vs few-shot vs financial-context prompting? | 4, 8 |
| RQ4 | How consistent are LLM predictions? | 6 |
| RQ5 | Can LLM weak labels train a smaller model? | 5 |
| RQ6 | Is Reddit sentiment associated with stock returns? | 7 |
| RQ7 | Does sentiment add predictive information? | 7 |
| RQ8 | Limitations and risks of LLMs for financial sentiment? | 8, 10 |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| No API keys for commercial LLMs | High | Use open-source models via Ollama or HuggingFace Inference API |
| CPU-only slows FinBERT/distillation | Medium | Small batch sizes, caching, subset sampling |
| Public Reddit dataset lacks quality labels | Medium | Create human-annotated test set; use LLM labels as weak supervision |
| Dataset too small | Medium | Combine multiple Kaggle/HuggingFace sources |
| No suitable timestamp data for market analysis | High | Select dataset with timestamps; fall back to synthetic alignment |
| LLM API costs | Medium | Start with 100-post debug runs; aggressive caching |
| Package compatibility (Python 3.14) | Low | Test each install; use compatible versions |

---

## Installation Requirements (to be installed)

```bash
pip install vaderSentiment yfinance seaborn pytest xgboost datasets
```

---

## Timeline Estimate

| Phase | Estimated Effort | Priority |
|-------|-----------------|----------|
| Phase 0 | Complete | --- |
| Phase 1 | 2-3 hours | Critical |
| Phase 2 | 1-2 hours | High |
| Phase 3 | 1-2 hours | High |
| Phase 4 | 3-4 hours | High |
| Phase 5 | 2-3 hours | High |
| Phase 6 | 1-2 hours | Medium |
| Phase 7 | 2-3 hours | High |
| Phase 8 | 1-2 hours | High |
| Phase 9 | 2-3 hours | Medium |
| Phase 10 | 1-2 hours | Critical |

---

*Plan created: 2026-09-25*
*Last updated: 2026-09-25*
