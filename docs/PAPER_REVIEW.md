# Paper Review: "What do LLMs Know about Financial Markets?"
## A Case Study on Reddit Market Sentiment Analysis

**Authors:** Xiang Deng, Vasilisa Bashlovkina, Feng Han, Simon Baumgartner, Michael Bendersky  
**Venue:** Companion Proceedings of the ACM Web Conference 2023 (WWW '23)  
**arXiv:** [2212.11311](https://arxiv.org/abs/2212.11311)  
**Affiliations:** Ohio State University, Google

---

## 1. Problem

Market sentiment analysis on social media content requires knowledge of both financial
markets and social media jargon. This makes annotation challenging — even human experts
agree only ~70% of the time. The scarcity of high-quality labeled data for Reddit financial
posts blocks conventional supervised learning approaches.

**Key challenge:** Reddit posts are noisy, contain slang ("tendies", "diamond hands", "to the
moon"), mix multiple arguments, and often use sarcasm or irony. Unlike Twitter/Stocktwits,
Reddit posts can be very long (due diligence posts) and cover a broader range of topics.

## 2. Methodology

The paper proposes a **semi-supervised learning pipeline** with two stages:

### Stage 1: In-Context Learning with LLM (Weak Label Generation)

1. **LLM used:** PaLM-540B
2. **Prompt design:**
   - Task description (domain-adapted: stock price movement expectation as proxy for financial sentiment)
   - 6 demonstration examples (2 per sentiment class: bullish, bearish, neutral)
   - Manually written Chain-of-Thought (CoT) reasoning in demonstrations
3. **Inference strategy:**
   - Temperature sampling (T=0.5) instead of greedy decoding
   - 8 repeated generations per input (multiple reasoning paths)
   - Majority voting for final label aggregation
4. **Key insight:** CoT forces the LLM to summarize the author's finance-related arguments
   (TL;DR-style) before drawing a conclusion, improving label stability.

### Stage 2: Knowledge Distillation (Student Model Training)

1. **Student model:** Charformer (character-level T5, ~102M parameters)
   - Further pre-trained on social media content
2. **Distillation approach:**
   - Filter weakly-labeled examples: keep only those where LLM made ≥5 consistent
     predictions out of 8 reasoning paths
   - **Regression loss** instead of classification loss
   - Soft score = agreement ratio between multiple labels (not hard categorical)
   - This produces smoother precision-recall curves
3. **Training data:** ~17K Reddit posts (filtered from 20K initial samples)
4. **Hyperparameters:** Learning rate 1e-4, batch size 64, regression head on encoder
   (decoder dropped)

## 3. Dataset

### Reddit Dataset (Primary — Internal/Proprietary)
| Aspect | Detail |
|--------|--------|
| Source | Reddit posts |
| Topic filter | Proprietary finance topic classifier |
| Stock filter | Popularity-based (internal system) |
| Training set | 20,000 posts randomly sampled → ~17K after filtering |
| Test set | 100 posts, annotated by 3 in-house experts |
| Labels | Bullish / Neutral / Bearish (3-class) |
| Subreddits | Not explicitly listed (implied r/wallstreetbets and others) |
| Time period | Not explicitly specified |

### FiQA Benchmark (External Evaluation)
| Aspect | Detail |
|--------|--------|
| Source | Maia et al. (2018) |
| Subtasks | FiQA-News (news headlines), FiQA-Post (Twitter/Stocktwits microblogs) |
| Conversion | Original continuous scores → binary (positive/negative) |
| Removal | Sentiment score = 0, posts mentioning multiple stocks |
| Split | 80/10/10 (train/val/test) from original training set |

### Key Limitation
> **The Reddit dataset is PROPRIETARY and NOT publicly available.** The paper's
> Reddit data was collected and labeled using internal Google systems. We cannot
> reproduce this exact dataset.

## 4. Models

### Baselines
| Model | Type | Parameters | Notes |
|-------|------|------------|-------|
| FinBERT-ProsusAI | Fine-tuned BERT | 110M | `ProsusAI/finbert` on HuggingFace |
| FinBERT-HKUST | Fine-tuned BERT | 110M | `yiyanghkust/finbert-tone` on HuggingFace |
| Charformer (CF) fine-tuned on FiQA | Character-level T5 | 102M | Backbone model |

### LLM
| Model | Parameters | Usage |
|-------|------------|-------|
| PaLM-540B | 540B | In-context learning, weak label generation |
| PaLM-62B | 62B | Ablation study |

### Final Model
| Model | Description |
|-------|-------------|
| CF - Distilled PaLM | Charformer trained on PaLM-generated weak labels with regression loss |

## 5. Experiments

### Experiment 1: Direct LLM In-Context Learning
- PaLM with 6-shot + CoT + 8 reasoning paths + majority vote
- Evaluated on Reddit test set and FiQA

### Experiment 2: Ablation on Prompting
- Base prompt (no CoT) vs CoT
- Effect of shuffling demonstration order
- Number of reasoning paths (1 to 8)
- Model size comparison (62B vs 540B)

### Experiment 3: Distillation Methods
- Filtering threshold ablation (minimum consistent predictions: 5, 6, 7, 8 out of 8)
- Classification loss vs Regression loss
- Precision-recall curve comparison

### Experiment 4: Cross-Dataset Generalization
- Model trained on Reddit data → evaluated on FiQA (News + Post)

## 6. Results

### Main Results (Table 2 from paper — Accuracy)

> **IMPORTANT NOTE:** We report the paper's claimed results below. We have not
> independently verified these numbers. Our reproduction will generate its own metrics.

| Model | Reddit | FiQA-News | FiQA-Post |
|-------|--------|-----------|-----------|
| FinBERT-ProsusAI | — | — | — |
| FinBERT-HKUST | — | — | — |
| CF - FiQA | — | — | — |
| PaLM (6-shot + CoT + vote) | — | — | — |
| CF - Distilled PaLM | — | — | — |

> **Note:** The exact numerical values from Table 2 are not reliably extractable from
> the HTML rendering. The paper states that:
> - The Reddit dataset is "more challenging" than FiQA
> - PaLM performs "very well" with only 6 demonstrations
> - The distilled student model "outperforms all supervised baselines on Reddit"
> - The model "generalizes well to FiQA despite only being fine-tuned on Reddit"
> - The final model "performs on par with or better than existing state-of-the-art models"

### Key Findings
1. CoT reasoning significantly improves LLM label quality
2. Multiple reasoning paths + majority vote further stabilizes predictions
3. Regression loss outperforms classification loss for distillation
4. PaLM-540B benefits more from CoT than PaLM-62B
5. Filtering inconsistent labels improves student model quality
6. The final model generalizes across datasets (Reddit → FiQA)

### Error Analysis (from the paper)
- Majority of errors: neutral ↔ bullish or neutral ↔ bearish confusion
- Fewer positive/negative direct confusion errors
- Model struggles with: contradictory arguments, advanced investing actions
- Error rate: "wrong sentiment more than 30% of the time"

## 7. Limitations

1. **Proprietary dataset:** Reddit data labeled using internal Google topic classifier; not reproducible
2. **Proprietary LLM:** PaLM-540B is not publicly accessible
3. **Proprietary student model:** Charformer further pre-trained on internal social media data
4. **Small test set:** Only 100 human-annotated Reddit posts for evaluation
5. **No temporal analysis:** No investigation of sentiment-price relationships
6. **No ticker-level analysis:** Results are aggregate, not per-stock
7. **Inter-annotator agreement:** ~70% — meaning ground truth itself is noisy
8. **Ethical constraints acknowledged:** Model should not be used for investment decisions
9. **No open-source code/data release** (to our knowledge)

## 8. What We Reproduce

Since the exact paper methodology uses proprietary components (PaLM-540B, internal
Charformer, internal topic classifier, internal stock filter, internal Reddit dataset), we
reproduce the **spirit and methodology** of the paper, not a bit-for-bit replication:

| Paper Component | Our Reproduction |
|----------------|------------------|
| PaLM-540B | Available LLMs (GPT, Claude, or open-source models via Ollama) |
| Charformer student | DistilBERT / RoBERTa-base (small, servable models) |
| Internal Reddit dataset | Public Reddit financial datasets (Kaggle, HuggingFace) |
| Internal topic classifier | Subreddit-based filtering + ticker detection |
| 3-class sentiment | Same: Bullish / Neutral / Bearish |
| CoT + few-shot prompting | Reproduced faithfully |
| Multiple reasoning paths + voting | Reproduced faithfully |
| Regression loss distillation | Reproduced faithfully |
| Cross-dataset evaluation | Using FiQA (publicly available) |

### Key Differences We Document
1. Different LLM (not PaLM-540B)
2. Different student model backbone (not Charformer)
3. Different Reddit dataset (public, not proprietary)
4. Different filtering criteria (subreddit-based, not proprietary topic classifier)
5. Potentially different time period
6. Larger human-annotated test set if feasible

## 9. What We Extend

Our project extends the original paper in several ways:

| Extension | Description |
|-----------|-------------|
| **Multiple LLM comparison** | Compare commercial + open-source + financial-domain LLMs |
| **Prompt strategy comparison** | Zero-shot, few-shot, CoT, financial-context, structured output |
| **Traditional baselines** | Add VADER, TF-IDF+LogReg, TF-IDF+SVM as lower bounds |
| **FinBERT as baseline** | Systematic comparison with FinBERT (as paper did) |
| **Financial market analysis** | Investigate sentiment ↔ stock return associations (not in paper) |
| **Daily sentiment aggregation** | Aggregate sentiment by ticker and date |
| **Predictive experiments** | Test whether sentiment adds predictive information |
| **Event analysis** | Analyze high-activity days |
| **Engagement weighting** | Test whether engagement-weighted sentiment differs |
| **Calibration analysis** | Evaluate confidence calibration of models |
| **Consistency analysis** | Test LLM prediction stability |
| **Comprehensive error analysis** | Categorized error types with examples |
| **Interactive dashboard** | Streamlit-based exploration tool |
| **Larger evaluation set** | Target 500-1500 human-annotated posts |

---

*Last updated: 2026-09-25*
*This review is based on arXiv:2212.11311v1 (HTML rendering). Some tables/figures from
the original PDF may not have been fully captured in the HTML version.*
