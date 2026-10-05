# Paper Methodology Mapping: Deng et al. (WWW '23 Companion)

**Reference Paper:**  
*"What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis"*  
(Also cited as *"LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models"*)  
Xiang Deng, Vasilisa Bashlovkina, Feng Han, Simon Baumgartner, Michael Bendersky  
*Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)*, pp. 326–330. [arXiv:2212.11311](https://arxiv.org/abs/2212.11311)

---

## 1. Executive Summary & Mapping Overview

This document provides a line-by-line, section-by-section mapping between the published paper's methodology and our reproducible open-source implementation. Following the project specification, we faithfully reproduce the mathematical, algorithmic, and architectural core of the paper while explicitly documenting necessary modernizations for components that relied on proprietary Google infrastructure.

| Paper Section & Concept | Original Paper Approach | Our Reproduction Implementation | Modernization / Deviation Rationale | Category |
|:---|:---|:---|:---|:---|
| **Problem Definition** (§1) | 3-way financial outlook sentiment on Reddit (Bullish, Neutral, Bearish) | 3-way financial outlook normalized to `positive`, `neutral`, `negative` (mapped to `BULLISH`, `NEUTRAL`, `BEARISH`) | Exact match; strictly separated from generic emotional sentiment | `[PAPER REPRODUCTION]` |
| **Teacher LLM** (§2, §3.1) | PaLM-540B (proprietary Google internal model) | Configurable LLM interface: OpenAI (`gpt-4o-mini`, `gpt-4o`), Gemini (`gemini-1.5-flash`), Anthropic (`claude-3-5-sonnet`), Local (Ollama `llama3`), and deterministic MockLLM | PaLM-540B is retired and inaccessible; modern LLMs provide equivalent or superior reasoning capabilities via unified API | `[MODERNIZATION]` |
| **Prompt Architecture** (§3.1) | Task description + 6 demonstrations (2 Bullish, 2 Neutral, 2 Bearish) + CoT reasoning | YAML-configurable prompt template (`configs/prompts/sentiment_prompt.yaml`) with exactly 6 demonstrations (2 per class) and TL;DR reasoning instructions | Modularized into external configuration files for strict versioning and reproducibility | `[PAPER REPRODUCTION]` |
| **Multi-Path Sampling** (§3.1) | $K=8$ independent reasoning paths, temperature $T=0.5$ | Configurable multi-path generator with $K=8$, $T=0.5$, full output tracking schema | Exact replication of decoding hyper-parameters | `[PAPER REPRODUCTION]` |
| **Direct LLM Aggregation** (§3.1) | Majority voting across the 8 paths | Plurality / majority voting with explicit deterministic tie-breaking strategies | Exact replication with transparent, auditable tie handling | `[PAPER REPRODUCTION]` |
| **Soft Agreement Score** (§3.2) | Continuous agreement ratio across reasoning paths | Calculation of `positive_count`, `neutral_count`, `negative_count`, `dominant_label`, `agreement_score`, and continuous sentiment target $s \in [-1.0, 1.0]$ saved in `weak_labels.csv` | Exact replication of continuous target generation | `[PAPER REPRODUCTION]` |
| **Consistency Filtering** (§3.2) | Filtering weakly labeled samples: keep posts with $\ge M$ consistent paths out of 8 ($M=5$ default) | Configurable consistency filter evaluating $M \in \{5, 6, 7, 8\}$ with retention reporting | Exact replication of paper's filtering mechanism | `[PAPER REPRODUCTION]` |
| **Student Model Backbone** (§3.2) | Charformer (102M character-level T5 pre-trained on social media) | Configurable compact encoder: DistilBERT-base-uncased (default, 66M), DeBERTa-v3-small (44M), or MiniLM (33M) | Charformer weights and specialized social media pre-training are proprietary / unmaintained in HuggingFace; modern compact transformers serve identical role | `[MODERNIZATION]` |
| **Distillation Objective** (§3.2) | Regression loss (MSE) on continuous soft agreement score vs. Categorical Cross-Entropy | Dual distillation pipeline: (A) Categorical Cross-Entropy baseline, (B) MSE Regression loss on soft scores, followed by validation threshold search | Faithful reproduction of regression distillation core hypothesis | `[PAPER REPRODUCTION]` |
| **Student Hyperparameters** (§3.2) | Learning rate $1 \times 10^{-4}$, batch size 64 | Fully configurable via `configs/training.yaml`, default $lr=5 \times 10^{-5}$ adapted for modern sub-word transformers | Fine-tuned learning rate suitable for modern transformer backbones | `[MODERNIZATION]` |
| **Baselines** (§4.1) | FinBERT-ProsusAI, FinBERT-HKUST, Charformer (FiQA) | FinBERT-ProsusAI, FinBERT-HKUST, VADER, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, TF-IDF + Naive Bayes | Preserves paper baselines and adds classical lower-bound baselines | `[PAPER REPRODUCTION]` + `[PROJECT EXTENSION]` |
| **Cross-Dataset Benchmark** (§4.2) | FiQA 2018 (FiQA-News and FiQA-Post) binary conversion | Reproducible FiQA evaluation pipeline with score transformation ($>0 \to \text{pos}, <0 \to \text{neg}, =0 \to \text{discard}$) and multi-stock filtering | Exact replication of paper's external evaluation protocol | `[PAPER REPRODUCTION]` |
| **Financial Market Analysis** | Not present in original paper | Daily sentiment aggregation, 1/3/5-day forward returns, correlation (Pearson/Spearman), rolling windows, event analysis, OLS | Original paper stopped at NLP metrics; we extend to empirical market dynamics | `[PROJECT EXTENSION]` |
| **Interactive Dashboard** | Not present in original paper | 8-page Streamlit application covering live inference, model comparison, financial trends, error analysis, and ablations | Academic interface for reproducible exploration | `[PROJECT EXTENSION]` |

---

## 2. Detailed Methodological Breakdown

### Section 2.1: Problem Formulation (§1 of Paper)
- **Paper Formulation:** The task is framed as determining the author's financial outlook regarding a specific stock mentioned in a Reddit post. Sentiment is categorized into three discrete classes:
  1. **Bullish / Positive:** Expectation that the stock price will rise or positive assessment of corporate prospects.
  2. **Bearish / Negative:** Expectation that the stock price will drop or negative assessment of corporate prospects.
  3. **Neutral / Uncertain:** Objective reporting, factual questions, or ambiguous/balanced views without directional stance.
- **Implementation Mapping:** Handled by `src/llm/parser.py` and `src/data/preprocessing.py`. Labels are standardized internally to `positive`, `neutral`, and `negative`, while user-facing outputs and visualizers format them as `BULLISH`, `NEUTRAL`, and `BEARISH`.

### Section 2.2: Prompt Engineering & Chain-of-Thought Reasoning (§3.1 of Paper)
- **Paper Approach:** The paper demonstrates that standard zero-shot prompting fails on Reddit's noisy context. It uses 6-shot in-context learning with 2 examples per sentiment class. Each example contains:
  1. Reddit post content.
  2. A concise TL;DR / financial reasoning explanation highlighting key financial arguments and sentiment clues.
  3. The final sentiment label.
- **Implementation Mapping:** Stored in `configs/prompts/sentiment_prompt.yaml` and formatted via `src/llm/prompts.py`. The prompt strictly requires the LLM to output a JSON object or structured block:
  ```json
  {
    "reasoning_summary": "Concise 1-2 sentence financial rationale focusing on fundamental or technical arguments...",
    "sentiment": "positive|neutral|negative"
  }
  ```
  The strict parser (`src/llm/parser.py`) ensures that malformed outputs are repaired or retried, preventing pipeline failures.

### Section 2.3: Multiple Reasoning Paths & Majority Voting (§3.1 of Paper)
- **Paper Approach:** Sampling multiple reasoning paths ($K=8$) at temperature $T=0.5$ generates diverse reasoning chains that reduce stochastic hallucination. The direct LLM prediction is taken as the majority vote:
  $$\hat{y}_{\text{LLM}} = \arg\max_{c \in \{\text{pos}, \text{neu}, \text{neg}\}} \sum_{k=1}^K \mathbb{I}(y_k = c)$$
- **Implementation Mapping:** Implemented in `src/llm/labeling.py` and `src/llm/aggregation.py`. For each Reddit post:
  - Exactly 8 independent stochastic generations are performed.
  - Per-path metadata (`post_id`, `path_id`, `reasoning_summary`, `label`, `confidence`, `timestamp`) are persisted.
  - Plurality voting aggregates the 8 paths.
  - **Tie-Breaking Strategy:** In case of exact ties (e.g. 4 Bullish vs 4 Bearish, or 3-3-2), the tie-breaker follows a deterministic hierarchy: (1) Neutral default to represent uncertainty, or (2) selection based on the highest average token confidence if available, fully logged in metadata.

### Section 2.4: Soft Agreement Score & Consistency Filtering (§3.2 of Paper)
- **Paper Approach:** Converting multiple reasoning paths directly into a hard categorical label discards critical teacher confidence signals. The paper computes a soft score and filters out highly inconsistent posts:
  1. **Filtering:** Retain only posts where the dominant class count $C_{\max} \ge M$. The paper evaluates $M \in \{5, 6, 7, 8\}$ and selects $M=5$ (i.e. $\ge 5/8 \approx 62.5\%$ agreement) for the final training set.
  2. **Continuous Soft Target:** The paper constructs a continuous agreement target for regression distillation.
- **Implementation Mapping:** Implemented in `src/llm/aggregation.py` and `src/distillation/dataset.py`. The weak label dataset (`weak_labels.csv`) contains:
  - `positive_count`, `neutral_count`, `negative_count`
  - `dominant_label`, `dominant_count`
  - `agreement_score` $= \frac{\text{dominant\_count}}{K}$
  - `teacher_soft_score` $= \frac{\text{pos\_count} - \text{neg\_count}}{K} \in [-1.0, 1.0]$
  Posts failing the threshold ($C_{\max} < M$) are filtered out before student distillation.

### Section 2.5: Student Model & Regression Distillation (§3.2 of Paper)
- **Paper Approach:** 
  - Backbone: Charformer (102M character-level T5 pre-trained on internal Google social media dumps).
  - Objective: Regression loss (Mean Squared Error) between student continuous prediction and the teacher soft score:
    $$\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum_{i=1}^N (f_\theta(x_i) - s_i)^2$$
  - Inference: A threshold search on the validation set tunes cutoffs $(\theta_1, \theta_2)$ to map continuous student outputs back to discrete sentiment classes.
- **Implementation Mapping:** 
  - Backbone Modernization: Charformer is not supported in contemporary PyTorch/HuggingFace ecosystems. We use DistilBERT-base-uncased (66M parameters) as primary backbone, with configurable support for DeBERTa-v3-small and MiniLM in `src/distillation/student_model.py`.
  - Distillation Training (`src/distillation/train.py`):
    - **Experiment R1 (Classification Baseline):** CrossEntropyLoss on hard majority labels.
    - **Experiment R2 (Regression Distillation):** MSELoss on `teacher_soft_score`.
    - **Validation Thresholding (`src/distillation/evaluate.py`):** Grid search across $\theta \in [0.10, 0.40]$ optimizing validation Macro F1 to establish class boundaries:
      $$\hat{y} = \begin{cases} \text{BULLISH} & \text{if } \hat{s} > \theta \\ \text{BEARISH} & \text{if } \hat{s} < -\theta \\ \text{NEUTRAL} & \text{otherwise} \end{cases}$$

---

## 3. Evaluation Protocols & Baselines (§4 of Paper)

### Section 3.1: Benchmark Datasets
1. **Reddit Dataset (Primary):**
   - *Paper:* 20,000 internal Reddit posts filtered by internal topic classifiers, evaluated on 100 expert-annotated posts.
   - *Our Implementation:* Standardized public Reddit financial posts dataset with verified timestamps, tickers, and quality metrics (`data/raw/reddit_posts.parquet` and `data/processed/processed_reddit.parquet`), partitioned chronologically or stratified into Train, Validation, and Test splits.
2. **FiQA Benchmark (Cross-Dataset Generalization):**
   - *Paper:* FiQA-2018 Task 1 (headlines and microblogs). Continuous scores in $[-1, 1]$ transformed to binary: $>0 \to \text{positive}$, $<0 \to \text{negative}$, $0 \to \text{discarded}$. Multi-stock posts removed.
   - *Our Implementation:* FiQA evaluation protocol implemented in `src/data/ingestion.py` and `src/evaluation/benchmark.py` following the exact transformation logic.

### Section 3.2: Baseline Models
- **VADER:** Rule-based lexicon baseline adapted with financial domain terms.
- **TF-IDF + ML:** Logistic Regression, Linear SVM, and Multinomial Naive Bayes.
- **FinBERT-ProsusAI:** `ProsusAI/finbert` (Araci, 2019) fine-tuned on Financial PhraseBank.
- **FinBERT-HKUST:** `yiyanghkust/finbert-tone` (Yang et al., 2020) fine-tuned on corporate disclosures.
- **LLM Teacher Variants:**
  - Zero-shot
  - Few-shot (6-shot)
  - Few-shot + CoT reasoning
  - Few-shot + CoT + 8 reasoning paths + Majority Voting

---

## 4. Summary of Modernizations & Architectural Decisions

1. **Teacher LLM:** Decoupled through an abstract provider pattern (`BaseLLMProvider`) supporting OpenAI, Google Gemini, Anthropic Claude, Local Ollama, and a fully deterministic offline `MockLLMProvider` for zero-cost testing.
2. **Student Model:** Replaced proprietary Charformer with modern, accessible sub-word transformer encoders (DistilBERT / DeBERTa) that achieve superior token-level efficiency and reproducible training.
3. **Data Integrity:** Strict chronological and stratified train/val/test splitting to prevent future data leakage into financial evaluation.
4. **Reproducibility:** Pinned seeds (`seed=42`), centralized YAML configurations, cached API responses, and comprehensive unit and smoke tests.
