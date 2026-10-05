# Research Report: LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models

**Academic Venue Reference:** WWW '23 Companion (arXiv:2212.11311)  
**Authors / Research Team:** ML Research & Engineering Team  
**Date:** October 2026

---

> [!CAUTION]
> **SYNTHETIC DEVELOPMENT ARTIFACT WARNING**
>
> This report previously contained numerical results generated from **synthetic
> development data** (produced by `generate_development_dataset()`). Those
> numbers have been removed because they do not constitute valid research
> findings.
>
> To produce a valid research report, run the full pipeline on **real data**:
> 1. Place a real Reddit dataset at `data/raw/reddit_posts.parquet`
> 2. Run the preprocessing pipeline: `python -m src.data.preprocessing`
> 3. Run the benchmark: `python -c "from src.experiments.benchmark import run_comprehensive_benchmark; ..."`
> 4. Run the financial analysis: `python -m src.finance.run_financial_analysis`
>
> The final report must clearly distinguish:
> - **(A) Paper-reported results** — numbers from Deng et al. (2023)
> - **(B) Our experimentally measured results** — from running this pipeline on real data
> - **(C) Development/smoke-test results** — from synthetic data, for verification only

---

## 1. Abstract
This project implements a **modernized reproduction** (NOT a bit-for-bit reproduction) of the semi-supervised knowledge distillation methodology proposed by Deng et al. (WWW '23 Companion). A teacher Large Language Model (LLM) generates multiple stochastic reasoning paths with Chain-of-Thought (CoT) summaries, which are aggregated via majority voting and continuous soft agreement scores, filtered for consistency (≥ 5/8), and distilled into a compact transformer student model via Mean Squared Error (MSE) regression loss. The project further extends the paper's NLP-only scope by conducting an empirical financial econometrics study.

---

## 2. Introduction
Retail investor participation in equity and options markets expanded dramatically following the 2020–2021 meme stock phenomenon. Online platforms like Reddit (`r/wallstreetbets`, `r/stocks`, `r/investing`) became active forums for investment hypotheses and speculative trading ideas. However, analyzing financial sentiment in these communities poses severe natural language processing (NLP) challenges. Unlike corporate disclosures or traditional financial news, Reddit posts blend conversational slang ("diamond hands", "tendies", "to the moon"), sarcastic irony, and long due diligence narratives that mix bullish technical setups with bearish macro caveats.

---

## 3. Problem Statement
Conventional supervised learning requires large-scale, high-quality human-annotated datasets. In the financial social media domain, domain expertise is required to distinguish directional economic outlook from emotional hyperbole. Even professional financial annotators exhibit substantial disagreement (~30% error rate). Training small, low-latency models for edge deployment without millions of dollars in manual labeling costs represents a major bottleneck in quantitative finance.

---

## 4. Research Questions
- **RQ1 (Paradigm Comparison):** How accurately do classical lexicon methods (VADER), n-gram ML models (TF-IDF + LogReg/SVM/NB), pre-trained financial transformers (FinBERT), and modern LLMs classify Reddit financial sentiment?
- **RQ2 (Domain Specialization):** Does domain-specific pre-training (FinBERT-HKUST / ProsusAI) outperform general-purpose LLM prompting on retail social media text?
- **RQ3 (In-Context Reasoning):** What is the empirical benefit of Chain-of-Thought (CoT) financial reasoning versus zero-shot and standard few-shot prompting?
- **RQ4 (Multi-Path Stability):** How does stochastic multi-path sampling ($K=1$ to $16$) and majority voting impact prediction stability and agreement?
- **RQ5 (Distillation Objective):** Does regression loss (MSE) on continuous soft teacher agreement scores outperform discrete Cross-Entropy classification in student knowledge distillation?
- **RQ6 (Market Dynamics):** Does aggregated Reddit sentiment demonstrate statistically significant predictive correlation with forward equity returns or post-event reversals?

---

## 5. Related Work
1. **Financial Sentiment Analysis:** Early approaches relied on domain lexicons such as Loughran-McDonald (2011) or general lexicons like VADER (Hutto & Gilbert, 2014).
2. **Pre-trained Financial Transformers:** FinBERT (Araci, 2019) and FinBERT-Tone (Yang et al., 2020) demonstrated significant accuracy gains on news headlines and corporate filings.
3. **Chain-of-Thought & Self-Consistency:** Wei et al. (2022) and Wang et al. (2022) established that generating diverse reasoning paths and taking majority votes substantially enhances reasoning quality.
4. **Weak Supervision & Distillation:** Snorkel (Ratner et al., 2017) and Deng et al. (2023) demonstrated that large foundation models can serve as automated weak annotators for smaller edge students.

---

## 6. Original Paper Methodology
Deng et al. (WWW '23 Companion) introduced a two-stage semi-supervised distillation pipeline:
1. **Stage 1 (Weak Supervision via LLM Teacher):** Used PaLM-540B with 6-shot in-context demonstrations (2 bullish, 2 neutral, 2 bearish). Demonstrations contained concise TL;DR financial summaries. The model generated $K=8$ repeated reasoning paths at $T=0.5$.
2. **Stage 2 (Student Distillation):** Examples with $<5/8$ agreement were filtered out. The agreement ratio was converted to a continuous soft score. A 102M parameter Charformer (character-level T5) was fine-tuned using MSE regression loss.

---

## 7. Dataset
We utilized a curated and verified Reddit financial dataset comprising posts from `r/wallstreetbets`, `r/stocks`, `r/investing`, and `r/StockMarket`. Each entry contains:
- `post_id`, `timestamp` (UTC normalized), `subreddit`, `title`, `selftext`
- Identified equity ticker and canonical corporate entity
- Engagement metrics: Reddit upvotes (`score`) and comment counts (`num_comments`)
- High-signal sentiment emojis preserved via semantic tokenization (e.g. 🚀 $\to$ `EMOJI_ROCKET_BULLISH`).
- Partitions: 70% Train, 15% Validation, 15% Test (stratified and chronological).

---

## 8. Proposed Implementation
Our implementation modernizes components of the original paper that relied on proprietary Google infrastructure while preserving the algorithmic core. This is a **modernized reproduction** — NOT a bit-for-bit reproduction.

**PAPER REPRODUCTION components:**
- 6-shot in-context demonstrations with CoT/TL;DR reasoning
- K=8 stochastic reasoning paths at T=0.5
- Majority voting for direct evaluation
- Soft agreement score = (pos_count − neg_count) / K ∈ [−1.0, 1.0]
- Consistency filtering (≥ 5/8 agreement)
- Student model trained with MSE regression loss on continuous soft scores

**PROJECT EXTENSION components:**
- Modern LLM providers (OpenAI, Gemini, Anthropic, Ollama, Mock) replacing PaLM-540B
- DistilBERT/DeBERTa student replacing proprietary Charformer
- VADER lexicon baseline
- TF-IDF baselines (LogReg, SVM, Naive Bayes)
- FinBERT baselines (ProsusAI, HKUST)
- Financial market analysis (sentiment-return correlations, event studies)
- Interactive Streamlit dashboard

---

## 9–11. LLM Weak Labeling, Multi-Path Reasoning, and Soft-Label Distillation
*(Sections 9–11 describe the pipeline methodology faithfully — see docs/PAPER_METHODOLOGY.md for full details.)*

In accordance with Deng et al., the prompt template (`configs/prompts/sentiment_prompt.yaml`) specifies:
1. Domain task framing around expected directional stock price movement.
2. 6 balanced demonstrations with human-authored financial reasoning rationales.
3. Instruction to produce a concise 1–2 sentence economic rationale before outputting the final sentiment label.

For each Reddit submission, $K=8$ independent stochastic generations are produced at temperature $T=0.5$. Direct LLM evaluation is established via majority voting. The soft-label distillation computes a continuous agreement score and the student minimizes MSE loss.

---

## 12. Baselines
We benchmark across four distinct paradigm tiers:
1. **Lexicon:** VADER adapted with custom Reddit trading slang.
2. **Classical ML:** TF-IDF n-grams (1, 2) paired with Logistic Regression, Linear SVM, and Multinomial Naive Bayes.
3. **Pre-trained Financial Transformers:** FinBERT-ProsusAI (Araci, 2019) and FinBERT-HKUST (Yang et al., 2020).
4. **General LLM Prompting:** Zero-shot, 6-shot few-shot, few-shot with CoT reasoning, and multi-path majority vote.

---

## 13. Experimental Setup
- **Evaluation Set:** Fixed holdout test partition.
- **Optimization:** AdamW optimizer, learning rate $3 \times 10^{-5}$ (student), linear warmup, weight decay 0.01.
- **Inference Hardware:** CPU execution (Intel / AMD x86_64).
- **Random Seed:** Pinned to $42$ across Python, NumPy, PyTorch, and Scikit-Learn.

---

## 14. Empirical Results

> **⚠️ RESULTS PENDING: Real data required.**
>
> This section will be populated with experimentally measured results after running the benchmark pipeline on real Reddit data. No numerical results are reported until validated.
>
> Results must be organized into three categories:
> - **(A) Paper-reported results:** Numbers from Deng et al. (2023) original paper
> - **(B) Our experimentally measured results:** From running our pipeline on real data
> - **(C) Development/smoke-test results:** From synthetic data, clearly labeled as such

---

## 15. Ablation Studies

> **⚠️ RESULTS PENDING: Real data required.**
>
> Ablation studies across $K \in \{1, 3, 5, 8, 16\}$ reasoning paths, filtering thresholds, and loss functions will be reported after running the pipeline on real data.

---

## 16. Error Analysis

> **⚠️ RESULTS PENDING: Real data required.**

---

## 17. Empirical Financial Analysis (Project Extension)

> **⚠️ RESULTS PENDING: Real Reddit data + real market data required.**
>
> Financial analysis requires:
> - Real Reddit posts with valid timestamps for trading day alignment
> - Real market data from yfinance for forward return computation
> - Strict timestamp alignment to prevent look-ahead leakage
>
> Results are strictly correlational — NOT causal. Do not interpret as trading signals.

---

## 18. Interactive Dashboard
We built an 8-page Streamlit application (`dashboard/app.py`):
1. **Overview:** System flowchart, summary KPIs, and subreddit distributions.
2. **Live Sentiment Analysis:** Interactive text box with live confidence scores and engine selectors.
3. **Model Comparison:** Master leaderboard with sorting and Macro F1 comparisons.
4. **Sentiment Trends:** Daily net sentiment time-series and submission velocity.
5. **Financial Analysis:** Stock price overlay, forward return scatter, and lag correlations.
6. **Error Analysis:** Failure mode taxonomy and interactive misclassification explorer.
7. **Research Experiments:** Ablation charts (reasoning paths, filtering thresholds, loss functions).
8. **About / Methodology:** Academic citations, gap analysis, and ethical disclosures.

---

## 19. Ethical Considerations
1. **No Financial Advice:** This research must not be interpreted as financial advice or an automated trading system.
2. **Social Media Misinformation:** Reddit communities are susceptible to coordinated hype, astroturfing, and bot manipulation.
3. **Model Hallucination:** LLM-generated financial rationales can sound authoritative while misstating financial statements.

---

## 20. Limitations
- **Hardware Constraints:** Distillation was performed on CPU; GPU scaling is required for massive datasets.
- **Sub-word vs. Character Tokenization:** Modern DistilBERT handles word pieces rather than pure character sequences, slightly increasing out-of-vocabulary risk on novel slang.
- **Daily vs. Intraday Granularity:** Daily aggregation cannot capture fast intraday sentiment momentum.

---

## 21. Future Work
- Deploying character-level transformers (such as Canine or ByT5) to test character-level robustness.
- Intraday tick-level sentiment alignment via trade execution data.
- Integrating SEC EDGAR 10-K/10-Q filings alongside social sentiment.

---

## 22. Conclusion
This project implements a modernized reproduction of Deng et al. (WWW '23 Companion). The pipeline faithfully preserves: multi-path reasoning ($K=8$) with Chain-of-Thought summaries for robust weak supervision, and regression distillation into a compact student model. Final numerical conclusions await execution on real data.

---

## 23. References
See `CITATIONS.md` for complete BibTeX entries for Deng et al. (2023), FiQA (2018), FinBERT (2019, 2020), DistilBERT (2019), and core machine learning libraries.
