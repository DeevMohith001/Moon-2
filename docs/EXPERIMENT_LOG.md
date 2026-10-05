# Experiment Log: Verified Runs on Real Data

**Project:** Reddit Market Sentiment Analysis with Large Language Models  
**Execution Environment:** Windows 11, Python 3.12.10, PyTorch 2.8.0, Transformers 4.55.4  
**Date:** October 2, 2026  
**Guiding Principle:** Every number in this document corresponds to an actual run executed on real data.

---

## 1. Data Ingestion & Preprocessing

### Command Executed:
```bash
python -m src.data.preprocessing
```

### Execution Details:
- **Raw Data Source:** HuggingFace `emilpartow/reddit_finance_posts_apple-tesla-microsoft`
- **Subreddits Included:** `r/stocks`, `r/investing`, `r/wallstreetbets`, `r/StockMarket`, `r/options`, `r/dividends`, `r/Superstonk`
- **Raw Post Count:** 12,046
- **After Length Filtering ($\ge 15$ characters):** 12,008
- **After Deduplication:** 10,596 unique posts
- **Posts with Resolved Tickers:** 9,932 (93.7%)
- **Data Provenance:** Tagged as `REAL` across all rows; saved to `data/processed/processed_reddit.parquet`

### Split Generation:
- **Strategy:** Stratified / Random seed pinned to `42`
- **Train Split (`data/processed/train.parquet`):** 7,417 posts (70.0%)
- **Validation Split (`data/processed/val.parquet`):** 1,589 posts (15.0%)
- **Test Split (`data/processed/test.parquet`):** 1,590 posts (15.0%)

### Data Integrity & Leakage Verification:
```bash
python -c "
import pandas as pd
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')
print('ID Overlap (Train/Val):', len(set(train['post_id']).intersection(set(val['post_id']))))
print('ID Overlap (Train/Test):', len(set(train['post_id']).intersection(set(test['post_id']))))
print('ID Overlap (Val/Test):', len(set(val['post_id']).intersection(set(test['post_id']))))
print('Text Overlap (Train/Val):', len(set(train['text']).intersection(set(val['text']))))
print('Text Overlap (Train/Test):', len(set(train['text']).intersection(set(test['text']))))
print('Text Overlap (Val/Test):', len(set(val['text']).intersection(set(test['text']))))
"
```
**Outcome:** Zero overlap across post IDs and zero overlap across text content. No data leakage detected.

---

## 2. FiQA Benchmark Cross-Dataset Evaluation

### Command Executed:
```bash
python -m src.experiments.benchmark
```

### Protocol Alignment:
- **Source:** HuggingFace `pauri32/fiqa-2018`
- **Subtask Filtering:** `format == 'headline'` for FiQA-News; `format == 'post'` for FiQA-Post
- **Protocol Cleaning:** Dropped $score = 0$ (ambiguous); dropped multi-stock texts
- **Target Label:** Binary classification (`BULLISH` if $score > 0$, `BEARISH` if $score < 0$)
- **Decision Rule for 3-Class Models:** Binary forced choice ($P(bull) \ge P(bear)$) per paper protocol

### Dataset Statistics:
- **FiQA-News:** 364 train samples, 54 validation samples, 60 test samples (68% Bullish / 32% Bearish)
- **FiQA-Post:** 593 train samples, 45 validation samples, 81 test samples (67% Bullish / 33% Bearish)

### Verified Results:
```
========================================================================================
Dataset        Model               Category             Accuracy  Macro F1  Precision  Recall  Samples
========================================================================================
FiQA-News      VADER (Lexicon)     [PROJECT EXTENSION]  50.00%    0.3273    0.3475     0.3457  60
FiQA-News      FinBERT (ProsusAI)  [PAPER REPRODUCTION] 75.00%    0.4996    0.5000     0.5017  60
FiQA-News      FinBERT (HKUST)     [PAPER REPRODUCTION] 61.67%    0.4052    0.4418     0.4254  60
FiQA-Post      VADER (Lexicon)     [PROJECT EXTENSION]  55.56%    0.3703    0.3703     0.3703  81
FiQA-Post      FinBERT (ProsusAI)  [PAPER REPRODUCTION] 71.60%    0.4769    0.4796     0.4778  81
FiQA-Post      FinBERT (HKUST)     [PAPER REPRODUCTION] 66.67%    0.4423    0.4476     0.4437  81
========================================================================================
```
*Artifacts Saved:* `results/fiqa_benchmark.csv` and `outputs/metrics/fiqa_benchmark.csv`.

---

## 3. Financial Market Return Econometric Analysis

### Command Executed:
```bash
python -m src.finance.run_financial_analysis
```

### Configuration & Data:
- **Target Equities:** AAPL, TSLA, MSFT, NVDA, GME, GOOGL, SPY, AMZN, META, AMD
- **Total Reddit Submissions:** 8,820 posts across target equities
- **Date Range:** 2020-01-01 to 2025-07-17 (active US trading calendar)
- **Price Engine:** `yfinance` daily OHLCV prices (1,391 trading days fetched per ticker)
- **Joint Ticker-Date Observations:** 3,083

### Econometric Results:
1. **Correlation with Forward Returns (Pearson $r$):**
   - $1$-Day Forward Return: $r = +0.0074$ ($p = 0.6823$, not statistically significant)
   - $3$-Day Forward Return: $r = -0.0080$ ($p = 0.6552$, not statistically significant)
   - $5$-Day Forward Return: $r = +0.0023$ ($p = 0.8966$, not statistically significant)
2. **Engagement Weighting:**
   - Log-weighted $1$-Day Return Correlation: $r = +0.0068$
3. **Event Analysis (Sentiment Spikes):**
   - $1$-Day Return Spread (Bullish Spikes vs Bearish Spikes): $-0.12\%$

*Artifacts Saved:*
- `data/processed/sentiment_returns_merged.parquet`
- `results/metrics/financial_analysis_results.json`
- `results/figures/sentiment_vs_return_scatter.png`

---

## 4. Test Suite Execution

### Command Executed:
```bash
pytest
```

### Results Summary:
- **Collected:** 28 items
- **Passed:** 28
- **Failed:** 0
- **Duration:** 32.27s
- **Breakdown:**
  - 5 Data Integrity & Provenance tests (`test_data_integrity.py`)
  - 7 Preprocessing & Ticker Detection tests (`test_preprocessing.py`)
  - 3 VADER Baseline tests (`test_vader.py`)
  - 2 Finance Alignment & Math tests (`test_finance.py`)
  - 1 Metrics Computation test (`test_metrics.py`)
  - 9 LLM Prompt, Parser & Multi-Path tests (`test_llm_pipeline.py`)
  - 1 End-to-End Smoke Distillation Pipeline test (`test_smoke_pipeline.py`)

---

## 5. Interactive Research Dashboard

### Command Executed:
```bash
streamlit run dashboard/app.py --server.headless=true --server.port=8501
```

### Verification Outcome:
- Dashboard initialized successfully and served on `http://localhost:8501`.
- Verified all 8 pages load with current, real artifacts:
  - Page 1: Overview & Architecture
  - Page 2: Interactive Sandbox (VADER, TF-IDF, LLM options)
  - Page 3: Model Comparison & FiQA Benchmark Leaderboard
  - Page 4: Sentiment Dynamics & Post Volume Velocity
  - Page 5: Financial Econometrics & Return Correlation Scatter Plots
  - Page 6: Error Analysis Taxonomy
  - Page 7: Research Experiments & Ablation Studies
  - Page 8: Methodology, Ethics & Paper Attributions
