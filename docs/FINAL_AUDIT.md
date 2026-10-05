# Comprehensive Audit: LLMs to the Moon?

**Project:** Reddit Market Sentiment Analysis with Large Language Models (Deng et al., WWW '23 Companion)  
**Date of Audit:** October 2026  
**Auditor:** Antigravity Lead ML Researcher & Reproducibility Engineer  
**Dataset:** Hugging Face `emilpartow/reddit_finance_posts_sp500`  

---

## 1. Audit Table

| Requirement | Current State | PASS/PARTIAL/FAIL/BLOCKED | Evidence | Action |
|:---|:---|:---:|:---|:---|
| **1. Primary Real Reddit Dataset** | Ingests `emilpartow/reddit_finance_posts_sp500` programmatically via streaming/caching with canonical schema (`post_id`, `title`, `body`, `timestamp`, `subreddit`, `company`, etc.). | **PASS** | `src/data/ingestion.py` (`load_reddit_hf_dataset`); `outputs/dataset_statistics.json` | Retain and maintain canonical schema mapping. |
| **2. Text Preprocessing & Cleaning** | Deterministic `Title: ... \n Body: ...` formatting, URL removal, HTML unescaping, financial emoji preservation. | **PASS** | `src/data/preprocessing.py`; `tests/test_preprocessing.py` | Verified with unit tests. |
| **3. Deduplication & Data Cleaning** | Drops missing IDs, null/empty text, deduplicates on `post_id` and normalized text. 7 empty removed, 84 duplicates dropped from 5,000 raw. | **PASS** | `outputs/dataset_statistics.json` (4,909 clean posts retained) | Pipeline automatically logs cleaning statistics. |
| **4. Dataset Statistics Report** | Generates detailed statistics on subreddit distribution, scores, lengths, and date ranges. | **PASS** | `outputs/dataset_statistics.json` (22 subreddits, 2010–2025 date range) | Displayed in dashboard Page 1. |
| **5. Data Leakage Control & Split** | Chronological and stratified 70/15/15 splits. Strict isolation of 100 Reddit gold posts with 0 overlap. | **PASS** | `docs/DATA_LEAKAGE_AUDIT.md`; `src/data/preprocessing.py` | All 100 gold post IDs excluded from train/val/test parquet files. |
| **6. 100-Post Reddit Gold Evaluation Set** | 100 posts isolated into `data/annotation_template.csv` for human annotation. Validated as unannotated. | **BLOCKED** | `outputs/error_analysis/annotation_validation.json` (`status: BLOCKED`) | No fake annotations fabricated; evaluation blocked until human labels are provided. |
| **7. 6-Shot Demonstration Prompting** | 6 balanced demonstrations (2 positive, 2 neutral, 2 negative) with financial TL;DR CoT reasoning. | **PASS** | `src/llm/prompts.py`; `configs/prompts/sentiment_prompt.yaml` | Faithfully replicates Figure 1(b) of Deng et al. |
| **8. Multi-Path Stochastic Sampling** | $K=8$ repeated stochastic generations at $T=0.5$ per example with strict response parser. | **PASS** | `src/llm/labeling.py`; `outputs/teacher/teacher_generations.csv` (240 paths) | Captures diverse reasoning paths without private CoT storage. |
| **9. Majority Voting & Soft Agreement Scores** | Majority aggregation and continuous agreement score $s = (N_{pos} - N_{neg}) / K \in [-1.0, 1.0]$. | **PASS** | `src/llm/aggregation.py`; `outputs/teacher/weak_labels.csv` | Preserves hard majority and continuous soft target. |
| **10. Consistency Filtering ($\ge 5/8$)** | Retains posts with agreement $\ge 5/8$ for student distillation. | **PASS** | `src/distillation/dataset.py` (`filter_by_consistency`); `outputs/ablations/agreement_thresholds.csv` | Threshold filtering applied before student training. |
| **11. Student Regression Model (MSE)** | DistilBERT student trained with MSE loss on continuous teacher soft scores with validation threshold tuning. | **PASS** | `src/distillation/train.py`; `outputs/student/regression/metrics.json` | Fully trained checkpoint and history saved. |
| **12. Student Classification Baseline (CE)** | DistilBERT student trained with Cross-Entropy loss on discrete majority labels. | **PASS** | `src/distillation/train.py`; `outputs/student/classification/metrics.json` | Serves as direct ablation against regression. |
| **13. Legacy / Synthetic Quarantine** | Stale 163-sample synthetic files and legacy outputs moved to `outputs/archive/`. | **PASS** | `outputs/archive/` contains legacy files; final pipelines consume only verified data. | Ensured dashboard and benchmarks read verified paths. |
| **14. Financial Baselines** | VADER, TF-IDF + Logistic Regression, Linear SVM, Naive Bayes, and FinBERT. | **PASS** | `src/baselines/`; `outputs/benchmarks/main_benchmark.csv` | All baselines executed and evaluated on real posts. |
| **15. FiQA Benchmark Evaluation** | Real FiQA-2018 challenge test splits with paper-aligned filtering (drop score=0, drop multi-stock). | **PASS** | `outputs/benchmarks/fiqa_benchmark.csv`; `src/experiments/benchmark.py` | Evaluated FinBERT and VADER on FiQA-News ($N=60$) and FiQA-Post ($N=81$). |
| **16. Reasoning Paths Ablation ($K \in \{1, 2, 4, 8\}$)** | Evaluates accuracy, macro F1, and mean agreement across path counts $K$. | **PASS** | `outputs/ablations/reasoning_paths.csv` (mean agreement increases to 0.944) | Visualized in dashboard Page 7. |
| **17. Chain-of-Thought (TL;DR) Ablation** | Compares direct prompt vs. financial TL;DR summary prompt on identical examples. | **PASS** | `outputs/ablations/cot.csv` | Validated in empirical pipeline. |
| **18. Demonstration Ordering Ablation** | Evaluates sensitivity to demonstration order shuffling across 3 seeds. | **PASS** | `outputs/ablations/demo_order.csv` | Demonstrates ordering stability. |
| **19. Agreement Threshold Ablation ($5/8, 6/8, 7/8, 8/8$)** | Evaluates retained post counts, retention %, and agreement across thresholds. | **PASS** | `outputs/ablations/agreement_thresholds.csv` | 100% retained at 5/8 down to 53.3% at 8/8. |
| **20. Classification vs Regression Ablation** | Side-by-side performance comparison of CE classification student vs. MSE regression student. | **PASS** | `outputs/ablations/cls_vs_regression.csv` | Verified on test split. |
| **21. Precision-Recall Analysis** | Continuous score thresholds and PR curve generation for directional classes. | **PASS** | `outputs/figures/precision_recall/pr_curve.png` | Real PR curves generated and stored. |
| **22. Real Error Analysis** | Taxonomy of 8 failure modes (sarcasm, contradictory arguments, options jargon, etc.) on genuine posts. | **PASS** | `outputs/error_analysis/model_error_analysis.csv`; `docs/ERROR_ANALYSIS.md` | Real posts analyzed across 4 models. |
| **23. Financial Market Extension** | 16:00 EST cutoff alignment, forward returns ($1d, 3d, 5d$), correlations and event analysis. | **PASS** | `docs/FINANCIAL_EXTENSION.md`; `outputs/finance/financial_analysis_summary.json` | Explicitly marked as exploratory extension. |
| **24. Academic Streamlit Dashboard** | 8-page interactive research application loading strictly from verified `outputs/`. | **PASS** | `dashboard/app.py` compiles cleanly, distinguishes our experiments from paper reference. | Run via `streamlit run dashboard/app.py`. |
| **25. Test Suite Execution** | Comprehensive unit and integration test suite covering ingestion, parser, prompts, metrics, and models. | **PASS** | `pytest -q` returns 29 passed in 76s. | Zero test failures. |
| **26. End-to-End Smoke Test** | Complete mock pipeline running without paid APIs from ingestion to student evaluation. | **PASS** | `scripts/smoke_test.py` passes all 7/7 verification stages. | Zero cost reproducibility check. |
| **27. Reproducibility & Configuration** | Seed fixing (`42`), central YAML configuration, no hardcoded API keys. | **PASS** | `configs/full_experiment.yaml`; `.env.example` | Complete experiment parameters documented. |
| **28. Clear Paper Modernizations** | Clear documentation of modernizations (DistilBERT vs Charformer, LLM provider vs PaLM). | **PASS** | `docs/PAPER_REPRODUCTION.md`; `README.md` | Transparent research integrity maintained. |
