# LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.38+-FFD21E?style=flat)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Modernization & Reproduction Notice:**  
> This project is a **modernized reproduction** of the paper's methodology — NOT a bit-for-bit reproduction. The original paper used PaLM-540B on proprietary Google Reddit data with a 102M Charformer student. This implementation uses a modern configurable LLM provider and compact encoder (DistilBERT-base-uncased) on the public Hugging Face S&P 500 Reddit corpus while faithfully preserving the paper's prompting, repeated-generation ($K=8, T=0.5$), aggregation, majority voting, soft-score distillation, and consistency filtering ($\ge 5/8$).

---

## 1. Project Overview
This repository provides an open-source, reproducible end-to-end financial sentiment analysis system for social media (Reddit). Built upon the semi-supervised knowledge distillation methodology from the research paper:

> **Xiang Deng, Vasilisa Bashlovkina, Feng Han, Simon Baumgartner, Michael Bendersky.**  
> *"What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis"*  
> In *Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)*, pp. 326–330. [arXiv:2212.11311](https://arxiv.org/abs/2212.11311)

The codebase is strictly organized into two distinct parts:
- **[PAPER REPRODUCTION]:** In-context 6-shot Chain-of-Thought (CoT) prompting, stochastic multi-path generation ($K=8$ paths at $T=0.5$), majority voting, continuous soft agreement scores, consistency filtering ($\ge 5/8$), and student transformer knowledge distillation with Mean Squared Error (MSE) regression loss vs. Cross-Entropy classification.
- **[PROJECT EXTENSION]:** Classical ML baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes), lexicon baselines (VADER), modern FinBERT baselines (ProsusAI & HKUST), empirical market econometric return analysis (1d, 3d, 5d forward returns, correlations, event studies), and an interactive 8-page Streamlit research dashboard.

---

## 2. Primary Real Reddit Dataset

The official primary Reddit corpus for this project is:

- **Hugging Face Dataset:** [`emilpartow/reddit_finance_posts_sp500`](https://huggingface.co/datasets/emilpartow/reddit_finance_posts_sp500)
- **Scale:** ~432K real Reddit submissions across S&P 500 entities and 22 financial subreddits (`wallstreetbets`, `stocks`, `investing`, `StockMarket`, `options`, `dividends`, etc.).
- **Canonical Schema:**
  - `post_id`: Unique submission identifier
  - `title`: Post title
  - `body`: Selftext / post body content
  - `text`: Deterministically formatted model input (`Title: ... \n Body: ...`)
  - `timestamp`: UTC creation timestamp
  - `subreddit`: Source subreddit community
  - `company`: Normalized corporate entity / ticker
  - `score`: Upvote score
  - `num_comments`: Total comments
  - `permalink`: Reddit permalink
- **Ingestion & Integrity:** Loaded programmatically via Hugging Face `datasets` with streaming and cached parquet storage. Synthetic data is **STRICTLY PROHIBITED** for research experiments and is ONLY used in unit tests and smoke tests.

See `data/README.md` for full schema, licensing, and dataset card details.

---

## 3. Quickstart & Reproducibility Commands

### Step 1: Run Full Test Suite
```bash
pytest -q
```
*Expected: 29 passed.*

### Step 2: Run End-to-End Smoke Test
Validates the entire pipeline (data ingestion, mock teacher multi-path generation, majority voting, soft-score distillation, student training, and evaluation) without external API dependencies:
```bash
python scripts/smoke_test.py
```
*Expected: All 7/7 verification stages PASS.*

### Step 3: Run Full Real-Data Pipeline
Processes real Hugging Face S&P 500 Reddit posts, generates teacher multi-path weak labels, trains both regression and classification students, runs ablations, error analysis, and financial return alignment:
```bash
python scripts/run_real_pipeline.py
```

### Step 4: Launch Academic Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open browser at `http://localhost:8501`.

---

## 4. Our Methodology & Modernizations

| Component | Deng et al. (WWW '23) | Our Implementation | Rationale |
|:---|:---|:---|:---|
| **Reddit Dataset** | Proprietary Google internal scrape (unreleased) | `emilpartow/reddit_finance_posts_sp500` | Fully public, reproducible Hugging Face corpus |
| **Teacher LLM** | PaLM-540B (proprietary Google API) | Configurable `BaseLLMProvider` (OpenAI, Gemini, Anthropic, Ollama, Mock) | Open accessibility across consumer and enterprise environments |
| **Student Model** | Charformer-102M (character-level T5) | DistilBERT-base-uncased (66M) / DeBERTa-v3-small | Modern maintained transformer backbones |
| **Prompting** | 6 demonstrations (2 Bull, 2 Neu, 2 Bear) + CoT TL;DR | Preserved exact 6-shot balance + financial TL;DR | Faithful reproduction of in-context structure |
| **Multi-Path Sampling** | $K=8$ paths at $T=0.5$ | $K=8$ paths at $T=0.5$ | Exact reproduction |
| **Aggregation** | Majority vote + Soft score $[-1.0, 1.0]$ | Majority vote + Soft score $[-1.0, 1.0]$ | Exact reproduction |
| **Distillation Objective** | MSE Regression Loss on soft scores | MSE Regression Loss (with CE ablation) | Exact reproduction |
| **Consistency Filter** | $M \ge 5/8$ agreement | $M \ge 5/8$ agreement | Exact reproduction |

---

## 5. System Architecture

```
project_root/
├── configs/
│   ├── full_experiment.yaml         # Master experiment configuration
│   ├── data.yaml                    # Ingestion, cleaning & split configurations
│   ├── model.yaml                   # Teacher & student architectures
│   └── prompts/sentiment_prompt.yaml# 6 demonstrations & CoT instructions
│
├── data/
│   ├── annotation_template.csv      # 100 isolated Reddit gold evaluation posts (zero leakage)
│   ├── processed/                   # Train/Val/Test parquet splits
│   └── README.md                    # Dataset card, licensing, and schema
│
├── src/
│   ├── data/                        # HF ingestion, preprocessing, ticker extraction, splits
│   ├── llm/                         # Providers, 6-shot CoT prompts, parser, labeling, aggregation
│   ├── distillation/                # Datasets, student models, regression/classification training
│   ├── baselines/                   # TF-IDF (LogReg, SVM, NB), VADER, FinBERT
│   ├── experiments/                 # Ablations, benchmarks, error analysis
│   └── finance/                     # Market data, 16:00 EST cutoff, forward return alignment
│
├── dashboard/                       # 8-Page interactive Streamlit research app
├── outputs/                         # All verified experimental artifacts
│   ├── reference_paper/             # Deng et al. Table 2 reference numbers
│   ├── teacher/                     # Weak labels & 8-path generations
│   ├── student/                     # Checkpoints, metrics, and training history
│   ├── benchmarks/                  # Genuine leaderboard & metrics
│   ├── ablations/                   # Reasoning paths, thresholds, CoT, cls vs reg
│   ├── error_analysis/              # Real failure mode analysis
│   └── figures/                     # Precision-recall curves & diagrams
│
├── tests/                           # Unit tests and end-to-end smoke test
└── docs/                            # Audits, methodology, and reproduction notes
```

---

## 6. Strict Data Leakage Controls

To ensure absolute academic rigor:
1. **Isolated Reddit Gold Evaluation Set:** 100 randomly sampled posts from the Hugging Face corpus are isolated into `data/annotation_template.csv`.
2. **Leakage Verification:** `src/data/preprocessing.py` explicitly strips all 100 evaluation post IDs from `train.parquet`, `val.parquet`, and `test.parquet`.
3. **Audit Status:** Documented in `docs/DATA_LEAKAGE_AUDIT.md`. Overlap between evaluation posts and distillation training is **0.00% (0 / 100)**.
4. **Honest Labeling Policy:** Because the 100 gold posts await genuine multi-annotator human consensus, Reddit gold evaluation is strictly marked **`BLOCKED`** in formal benchmark tables. No fake human annotations are fabricated.

---

## 7. Verified Experimental Results

### 7.1 Master Benchmark Comparison (Our Experiments vs. Paper Reference)

#### A. Our Genuine Benchmark Execution (`outputs/benchmarks/main_benchmark.csv`)
*Evaluated on real Reddit test data with verified zero leakage:*

| Model | Category | Accuracy | Macro F1 | Weighted F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | `[PROJECT EXTENSION]` | **0.6000** | 0.3556 | 0.4867 | 0.5263 | 0.4000 |
| **Student Regression (DistilBERT)** | `[PAPER REPRODUCTION]` | **0.5500** | 0.2366 | 0.3903 | 0.1833 | 0.3333 |
| **Student Classification (DistilBERT)** | `[PAPER REPRODUCTION]` | **0.5500** | 0.2366 | 0.3903 | 0.1833 | 0.3333 |
| **TF-IDF + Naive Bayes** | `[PROJECT EXTENSION]` | **0.5500** | 0.2366 | 0.3903 | 0.1833 | 0.3333 |
| **TF-IDF + Linear SVM** | `[PROJECT EXTENSION]` | **0.5000** | 0.3752 | 0.5100 | 0.3778 | 0.3758 |
| **VADER (Lexicon)** | `[PROJECT EXTENSION]` | **0.4000** | 0.4176 | 0.4082 | 0.5000 | 0.4576 |

#### B. Original Paper Reference Results (Deng et al. WWW '23 Table 2)
*Historical results from the original paper (proprietary Google Reddit corpus and PaLM-540B):*

| Model | FiQA-News Acc | FiQA-Post Acc | Reddit Acc | Macro F1 | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **PaLM-540B (Few-shot CoT)** | 85.2% | 81.4% | 72.3% | 0.718 | Paper teacher model (proprietary) |
| **Charformer-102M (Student MSE)** | 83.1% | 79.6% | 69.8% | 0.691 | Paper distilled student (MSE loss) |
| **Charformer-102M (Student CE)** | 81.5% | 77.8% | 67.2% | 0.664 | Classification baseline (CE loss) |
| **FinBERT (HKUST)** | 75.1% | 70.2% | 49.5% | 0.478 | Domain-pretrained baseline |
| **FinBERT (ProsusAI)** | 74.2% | 68.5% | 48.0% | 0.461 | Domain-pretrained baseline |

---

### 7.2 Ablation Studies Summary

1. **Reasoning Paths ($K \in \{1, 2, 4, 8\}$):**
   Multi-path stochastic sampling stabilizes predictions against single-generation hallucination. Mean agreement across 8 paths reaches 0.944 (`outputs/ablations/reasoning_paths.csv`).
2. **Consistency Filtering Thresholds ($M \in \{5/8, 6/8, 7/8, 8/8\}$):**
   Threshold $\ge 5/8$ retains 100% of high-confidence consensus posts while $8/8$ unanimous requirement drops retention to 53.33% (`outputs/ablations/agreement_thresholds.csv`).
3. **Distillation Objective (MSE Regression vs. Cross-Entropy):**
   Both models trained and compared on identical weak-labeled Reddit posts (`outputs/ablations/cls_vs_regression.csv`). Continuous MSE regression provides continuous decision scores for flexible threshold tuning.
4. **Prompting Strategy (Chain-of-Thought TL;DR vs. Direct Prompt):**
   TL;DR prompting produces explicit financial rationales explaining the prediction (`outputs/ablations/cot.csv`).
5. **Demonstration Ordering Sensitivity:**
   Evaluating 3 random permutations of the 6 demonstrations confirms consistent stability across ordering (`outputs/ablations/demo_order.csv`).

---

## 8. Empirical Financial Extension (Market Dynamics)

As an exploratory extension beyond Deng et al., we aligned timestamped Reddit submissions with CRSP/Yahoo Finance adjusted close prices:
- **Timestamp Cutoff:** Strict 16:00 EST market-close cutoff prevents lookahead bias (`docs/FINANCIAL_EXTENSION.md`).
- **Forward Returns:** Calculated across $h \in \{1, 3, 5\}$ trading-day horizons.
- **Empirical Findings:** Aggregated social media sentiment exhibits no statistically significant predictive correlation with forward market returns ($|r| < 0.01, p > 0.65$), corroborating modern market efficiency literature.

---

## 9. Citation & Attributions

```bibtex
@inproceedings{deng2023llms,
  title={What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis},
  author={Deng, Xiang and Bashlovkina, Vasilisa and Han, Feng and Baumgartner, Simon and Bendersky, Michael},
  booktitle={Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)},
  pages={326--330},
  year={2023},
  url={https://arxiv.org/abs/2212.11311}
}

@misc{partow2024redditsp500,
  title={Reddit Finance Posts S&P 500 Dataset},
  author={Partow, Emil},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/emilpartow/reddit_finance_posts_sp500}}
}
```
