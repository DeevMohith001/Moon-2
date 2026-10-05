# Data Leakage Audit

**Project:** LLMs to the Moon? Reddit Market Sentiment Analysis  
**Dataset:** emilpartow/reddit_finance_posts_sp500  
**Audit Date:** October 2026  

---

## Overview

This document provides an explicit audit of all data leakage risks in the pipeline and verifies that our split policy prevents contamination between pools.

---

## 1. Pool Definitions

| Pool | Purpose | Source | Seed | Max Size |
|---|---|---|---|---|
| **Distillation Pool** | Teacher labeling + student training | sp500 HF dataset | 42 | 20,000 |
| **Reddit Gold Pool** | Human annotation + held-out evaluation | sp500 HF dataset | 9999 | 100 |
| **FiQA Test Pool** | Cross-dataset benchmark evaluation | pauri32/fiqa-2018 | N/A | ~141 |

---

## 2. Leakage Checks

### 2.1 Reddit Gold Set vs Distillation Pool

**Requirement:** The 100 evaluation posts must never appear in the distillation pool.

**Implementation:**
```python
# In scripts/create_annotation_set.py:
# 1. Load full dataset
# 2. Determine distillation post IDs (seed=42, max=20000)
# 3. Select annotation posts from REMAINING posts NOT in distillation pool (seed=9999)
# 4. Verify overlap == 0

overlap = set(annotation_df["post_id"]) & distillation_post_ids
assert len(overlap) == 0  # HARD FAIL if any overlap
```

**Status:**
- ✅ `VERIFIED` by `scripts/create_annotation_set.py` leakage check
- ✅ Zero post ID overlap enforced via set difference
- ✅ Different seeds (42 vs 9999) for reproducible separation
- Verified by: `scripts/create_annotation_set.py` (hard fails on any overlap)

### 2.2 FiQA Test Labels NOT in Student Training

**Requirement:** FiQA test split labels must not be used for student training.

**Implementation:**
- Student trains only on the Distillation Pool (Reddit sp500 posts)
- FiQA is loaded entirely separately via `load_fiqa_dataset()`
- FiQA test split is used ONLY in `run_fiqa_benchmark()` for evaluation
- FiQA labels are NEVER merged into the training DataFrame

**Status:**
- ✅ `VERIFIED` — separate code paths, no data merge between Reddit training and FiQA
- FiQA evaluation is in `src/experiments/benchmark.py::run_fiqa_benchmark()`
- Reddit training is in `src/distillation/train.py`
- No shared DataFrame operations

### 2.3 Demonstrations NOT in Evaluation Set

**Requirement:** The 6 in-context demonstration examples in the prompt must not appear in the evaluation set.

**Implementation:**
- Demonstrations are hand-crafted examples stored in `configs/prompts/sentiment_prompt.yaml`
- They are NOT from the sp500 dataset
- They are synthetic illustrative examples for the teacher prompt
- The evaluation set consists of sp500 posts with real Reddit content

**Status:**
- ✅ `VERIFIED` — demonstrations are hand-crafted, not from the HuggingFace corpus
- Evidence: `configs/prompts/sentiment_prompt.yaml` (hand-crafted examples)

### 2.4 Test Labels NOT in Student Training

**Requirement:** The test split of the Reddit corpus must not be used during student training.

**Implementation:**
```python
# In src/distillation/train.py:
# Training uses train_df (from train split)
# Validation uses val_df (from val split)
# Test evaluation is SEPARATE in evaluate_regression_student()
```

**Status:**
- ✅ `VERIFIED` by test `test_create_splits_zero_leakage_and_mutually_exclusive` in `tests/test_data_integrity.py`
- Post IDs checked for mutual exclusivity across train/val/test splits

### 2.5 No Future Financial Information in Historical Sentiment

**Requirement:** Sentiment labels for a post at time T must not use price data from time T+1 or later.

**Implementation:**
- Sentiment is labeled by the teacher LLM based ONLY on the post text content
- The LLM has no access to historical price data during labeling
- Financial return calculation in `src/finance/` uses forward returns computed AFTER the post timestamp
- The financial extension is explicitly marked as an "extension" not used for training

**Status:**
- ✅ `VERIFIED` — LLM labels are text-only
- ✅ Financial returns are computed post-hoc for correlation analysis only
- ✅ Forward returns are NOT used as training targets

### 2.6 Teacher Soft Score Source

**Requirement:** The student's training target (`teacher_soft_score`) must come from LLM multi-path predictions, NEVER from ground-truth labels.

**Implementation:**
```python
# In src/distillation/train.py::train_regression_student():
if "teacher_soft_score" not in train_df.columns:
    raise ValueError(
        "train_df must contain 'teacher_soft_score' column. "
        "This column must be computed from LLM multi-path predictions..."
    )
```

**Status:**
- ✅ `VERIFIED` — hard validation guard in training code
- `teacher_soft_score` is computed as `(positive_count - negative_count) / K`
- It is derived from 8 independent teacher LLM generations

---

## 3. Verification Test Suite

The following tests in `tests/test_data_integrity.py` verify leakage prevention:

| Test | Checks |
|---|---|
| `test_assert_real_data_rejects_synthetic` | Data provenance guard works |
| `test_assert_real_data_accepts_real` | Real data passes guard |
| `test_create_splits_zero_leakage_and_mutually_exclusive` | Train/Val/Test have zero post ID overlap |
| `test_fiqa_loader_protocol_and_subtasks` | FiQA filtering is correct |
| `test_standardize_hf_reddit_schema` | Schema normalization preserves provenance |

---

## 4. Temporal Leakage Analysis

| Risk | Details | Status |
|---|---|---|
| Train posts after test posts | Chronological split prevents future data in training | ✅ `VERIFIED` (chronological split) |
| Financial returns during training | Returns only computed in finance extension, NOT as training targets | ✅ `VERIFIED` |
| Teacher model knowledge | Teacher LLM was pre-trained on internet data including financial information; this is expected and documented in the paper | ⚠️ `ACKNOWLEDGED` — inherent to methodology |
| LLM cache reuse | LLM responses are cached by content hash; same post always gets same label | ✅ `ACCEPTABLE` — deterministic, intended |

---

## 5. Annotation Leakage Risk

The 100-post Reddit gold evaluation set has NO human labels until annotation is performed.

**Current Status:** `BLOCKED — Awaiting human annotations`

Until genuine human annotations are provided:
- The Reddit gold evaluation is `BLOCKED`
- The annotation template is available at `data/annotation_template.csv`
- The leakage check (zero overlap with distillation pool) is already verified
- Guidelines are in `docs/ANNOTATION_GUIDELINES.md`

**DO NOT FABRICATE** human annotations to unblock this step.

---

## 6. Summary

| Leakage Check | Status |
|---|---|
| 100 evaluation posts NOT in distillation data | ✅ `VERIFIED` |
| No test labels in student training | ✅ `VERIFIED` |
| No FiQA test labels in student training | ✅ `VERIFIED` |
| Demonstrations not leaked into evaluation | ✅ `VERIFIED` |
| No future financial information in historical sentiment | ✅ `VERIFIED` |
| Teacher soft score from LLM, not ground truth | ✅ `VERIFIED` |
| Reddit gold evaluation with genuine annotations | ⚠️ `BLOCKED` |
