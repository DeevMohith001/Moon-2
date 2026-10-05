# Distillation Method Documentation

**Paper:** Deng et al. (2023) — "LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models"  
**Implementation:** Methodology Reproduction with publicly reproducible dataset and modern LLM

---

## Overview

This document describes the exact distillation methodology implemented in this repository, how it maps to the original paper, and where deviations exist.

---

## 1. Teacher LLM Weak Labeling

### 1.1 Prompt Design

The teacher LLM receives a **six-shot in-context prompt** with Chain-of-Thought reasoning:

- **2 Positive (Bullish)** demonstration examples
- **2 Neutral** demonstration examples  
- **2 Negative (Bearish)** demonstration examples

Each demonstration includes:
1. The post text
2. A concise **Financial TL;DR / reasoning summary** (paper calls this "financial summary")
3. The final sentiment label (`positive`, `neutral`, or `negative`)

Format per demonstration:
```
Example N:
Ticker: {ticker}
Post: """{post_text}"""
Output:
{
  "reasoning_summary": "...",
  "sentiment": "positive|neutral|negative"
}
```

### 1.2 Multi-Path Sampling

**Paper specification:**
- K = 8 independent reasoning paths per post
- Temperature T = 0.5

**Implementation:**
```python
# src/llm/labeling.py
for k in range(num_paths=8):
    path_prompt = f"{prompt}\n\n[Sampling Path {k+1}/{num_paths}]"
    response = provider.generate(path_prompt)  # T=0.5 from provider config
```

Each path produces:
- `reasoning_summary`: Financial TL;DR
- `label`: One of `positive`, `neutral`, `negative`

### 1.3 Output Schema

Every generation is stored with these required columns:

| Column | Type | Description |
|---|---|---|
| `post_id` | string | Reddit post identifier |
| `path_id` | int | Reasoning path number (1–8) |
| `reasoning_summary` | string | Financial TL;DR |
| `label` | string | `positive`, `neutral`, or `negative` |
| `display_label` | string | `BULLISH`, `NEUTRAL`, or `BEARISH` |
| `confidence` | float | Parser confidence (0.0–1.0) |
| `timestamp` | string | UTC ISO timestamp |

---

## 2. Aggregation and Soft Score Computation

### 2.1 Majority Voting

Given K=8 paths with labels $(l_1, l_2, ..., l_8)$:

$$\text{majority\_label} = \arg\max_{c \in \{+,-,0\}} \text{count}(l_i = c)$$

**Tie-breaking:** If multiple classes share the maximum count, the default is `neutral` (represents teacher uncertainty).

### 2.2 Agreement Score

$$\text{agreement\_score} = \frac{\max(\text{pos\_count}, \text{neu\_count}, \text{neg\_count})}{K}$$

Range: [1/K, 1.0] = [0.125, 1.0] for K=8.

### 2.3 Soft Continuous Score (Key Innovation)

The paper's key insight is using the continuous soft score as the distillation target instead of the hard majority label:

$$\text{teacher\_soft\_score} = \frac{N_{pos} - N_{neg}}{K} \in [-1.0, +1.0]$$

Where:
- $N_{pos}$ = count of `positive` paths out of 8
- $N_{neg}$ = count of `negative` paths out of 8
- $K = 8$ total paths

**Examples:**
- 8/8 positive → soft_score = +1.0
- 6/8 positive, 2/8 negative → soft_score = +0.5
- 4/8 neutral, 2/8 positive, 2/8 negative → soft_score = 0.0
- 8/8 negative → soft_score = -1.0

This continuous score encodes the **degree of certainty** and is more informative than a hard 3-class label.

---

## 3. Consistency Filtering

**Paper specification:** Keep only examples with ≥ 5/8 agreement.

**Implementation:**
```python
# src/distillation/dataset.py::filter_by_consistency()
def filter_by_consistency(df, threshold=5, total_paths=8):
    dominant_count = df["dominant_count"]
    retained = df[dominant_count >= threshold]
    return retained, stats
```

**Ablation:** The threshold is varied in `[5, 6, 7, 8]` to study its effect on dataset size and student quality.

| Threshold | Interpretation |
|---|---|
| 5/8 | Weak majority (any simple majority beyond tie) |
| 6/8 | Moderate consensus |
| 7/8 | Strong consensus |
| 8/8 | Unanimous agreement |

---

## 4. Student Model Training

### 4.1 Regression Distillation (Primary — Paper Method)

**Loss function:**
$$\mathcal{L}_{regression} = \text{MSE}(\hat{s}, s_{teacher})$$

Where:
- $\hat{s}$ = student's predicted soft score (scalar output, regression head)
- $s_{teacher}$ = teacher's soft score = $(N_{pos} - N_{neg}) / K$

**Critical:** The student trains on `teacher_soft_score`, NEVER on ground-truth human labels.

```python
# src/distillation/train.py::train_regression_student()
criterion = nn.MSELoss()
loss = criterion(outputs.logits, targets)  # targets = teacher_soft_score
```

### 4.2 Classification Baseline (Comparison)

For ablation comparison:
$$\mathcal{L}_{classification} = \text{CrossEntropy}(\hat{y}, y_{majority})$$

Where $y_{majority}$ is the teacher's majority vote label (3-class).

### 4.3 Student Architecture

**Original paper:** Charformer (102M parameter character-level T5)
- Unavailable in modern HuggingFace ecosystem
- Pre-trained on proprietary Google social media data

**This implementation:** `distilbert-base-uncased` (66M parameters)
- Publicly available on HuggingFace
- Token-level encoder (word-piece), not character-level
- Comparable parameter scale
- Documented deviation from the paper

**Training configuration (from `configs/full_experiment.yaml`):**
```yaml
student_backbone: "distilbert-base-uncased"
learning_rate: 3.0e-5
batch_size: 16
epochs: 5
max_length: 256
weight_decay: 0.01
```

---

## 5. Inference: Soft Score to Sentiment Label

The regression model outputs a continuous value $\hat{s} \in (-\infty, +\infty)$.

**Optimal threshold:** A threshold $\theta$ is tuned on the validation set:

$$\hat{y} = \begin{cases} \text{BULLISH} & \text{if } \hat{s} > \theta \\ \text{BEARISH} & \text{if } \hat{s} < -\theta \\ \text{NEUTRAL} & \text{otherwise} \end{cases}$$

The optimal $\theta$ is chosen by grid search on $\theta \in [-0.5, 0.5]$ to maximize validation accuracy.

---

## 6. Pipeline Summary

```
Reddit Post (title + body)
         ↓
6-Shot CoT Prompt (2+2+2 demonstrations)
         ↓
8 Independent Teacher LLM Generations (T=0.5)
         ↓
Parse each: {reasoning_summary, label}
         ↓
Count: pos_count, neu_count, neg_count
         ↓
Compute:
  majority_label = argmax(counts)
  agreement_score = max(counts) / 8
  teacher_soft_score = (pos_count - neg_count) / 8
         ↓
Filter: agreement >= threshold (default 5/8)
         ↓
Student Training:
  Classification: CrossEntropy(logits, majority_label)
  Regression: MSELoss(logit, teacher_soft_score)
         ↓
Student Inference:
  Score s -> BULLISH if s > θ, BEARISH if s < -θ, else NEUTRAL
```

---

## 7. Deviations from Original Paper

| Aspect | Original Paper | This Implementation |
|---|---|---|
| Teacher LLM | PaLM-540B (Google internal) | Configurable: OpenAI/Gemini/Anthropic/Ollama/Mock |
| Student architecture | Charformer 102M (character-level T5, Google internal) | DistilBERT-base-uncased (66M, publicly available) |
| Reddit corpus | Internal proprietary corpus (~38K posts) | emilpartow/reddit_finance_posts_sp500 (~432K public posts) |
| Expert test set | 100 posts with expert annotations (proprietary) | 100-post template from HF corpus (awaiting annotation) |
| CoT format | Financial summary before label | Financial TL;DR in JSON: `reasoning_summary` field |

---

## 8. Code Locations

| Component | File |
|---|---|
| Prompt construction | `src/llm/prompts.py`, `configs/prompts/sentiment_prompt.yaml` |
| Multi-path generation | `src/llm/labeling.py` |
| Aggregation & soft score | `src/llm/aggregation.py` |
| LLM response parsing | `src/llm/parser.py` |
| Consistency filtering | `src/distillation/dataset.py` |
| Student training (regression) | `src/distillation/train.py::train_regression_student()` |
| Student training (classification) | `src/distillation/train.py::train_classification_student()` |
| Student architecture | `src/distillation/student_model.py` |
| Evaluation | `src/distillation/evaluate.py` |
