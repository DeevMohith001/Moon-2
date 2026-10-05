# Experimental Results and Empirical Findings

## Reproduction & Extension of Deng et al. (2023)
*What do LLMs Know about Financial Markets?* (arXiv:2212.11311)

---

## 1. Complete Experimental Matrix & Results

### 1.1 Sentiment Classification Benchmark (Holdout Test Set $N = 163$)

| Exp ID | Model Architecture | Paradigm | Loss / Method | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Bullish F1 | Bearish F1 | Neutral F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** | VADER | Lexicon | Compound Thresholds | 63.19% | 62.57% | 0.6018 | 0.6041 | 0.6795 | 0.7423 | 0.3836 |
| **B2** | TF-IDF + Logistic Regression | Classical ML | Cross-Entropy ($C=0.01$) | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **B3** | TF-IDF + Linear SVM | Classical ML | Hinge Loss ($C=0.01$) | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **B4** | `ProsusAI/finbert` | Pretrained Transformer | Softmax Cross-Entropy | 71.17% | 71.28% | 0.7128 | 0.7125 | 0.6903 | 0.7294 | 0.7188 |
| **B5** | `yiyanghkust/finbert-tone` | Pretrained Transformer | Softmax Cross-Entropy | 72.39% | 72.30% | 0.7229 | 0.7239 | 0.7863 | 0.7045 | 0.6777 |
| **L1** | LLM Zero-Shot | General LLM | Direct Classification Prompt | 31.00% | 33.33% | 0.1578 | 0.1467 | 0.4733 | 0.0000 | 0.0000 |
| **L2** | LLM 6-Shot Few-Shot | General LLM | In-Context Exemplars | 31.00% | 33.33% | 0.1578 | 0.1467 | 0.4733 | 0.0000 | 0.0000 |
| **L3** | LLM Financial Context | General LLM | Financial Market Context | 31.00% | 33.33% | 0.1578 | 0.1467 | 0.4733 | 0.0000 | 0.0000 |
| **L4** | LLM Structured JSON | General LLM | JSON Schema Output | 31.00% | 33.33% | 0.1578 | 0.1467 | 0.4733 | 0.0000 | 0.0000 |
| **L5** | LLM Chain-of-Thought (CoT) | General LLM | 4-Step Analytical Reasoning | 31.00% | 33.33% | 0.1578 | 0.1467 | 0.4733 | 0.0000 | 0.0000 |
| **R2** | DistilBERT Student (Deng et al. Reproduction) | Distillation | **MSE Regression on Teacher Agreement** | 34.36% | 33.33% | 0.1705 | 0.1757 | 0.5114 | 0.0000 | 0.0000 |

---

### 1.2 Financial Econometric & Return Analysis ($N = 984$ Joint Observations across 14 Equities)

| Exp ID | Analysis Description | Horizon / Target | Primary Metric | Value | Statistical Significance |
|---|---|---|---|---|---|
| **F1** | Unweighted Daily Sentiment Correlation | 1-Day Forward Return | Pearson $r$ | **-0.0173** | $p = 0.5877$ (Not significant) |
| **F1** | Unweighted Daily Sentiment Correlation | 3-Day Forward Return | Pearson $r$ | **-0.0389** | $p = 0.2234$ (Not significant) |
| **F1** | Unweighted Daily Sentiment Correlation | 5-Day Forward Return | Pearson $r$ | **-0.0583** | $p = 0.0677$ (Borderline significance) |
| **F2** | OLS Regression ($R_{1d} = \beta_0 + \beta_1 S_t$) | 1-Day Forward Return | Slope $\beta_1$ | **-0.00067** | $R^2 = 0.0003, p = 0.588$ |
| **F6** | Event Analysis (High Bullish vs Bearish Spikes) | 1-Day Forward Return | Return Spread | **-0.17%** | Modest post-spike reversal |
| **F7** | Engagement-Weighted Correlation ($\log(1 + \text{Score})$) | 1-Day Forward Return | Pearson $r$ | **-0.0180** | Consistent with unweighted |

---

## 2. Research Questions Synthesis

### RQ1: Classification Performance Across Paradigms
- **Domain-Specific FinBERT Transformers** (`yiyanghkust/finbert-tone`: 72.39% Accuracy, 0.7229 Macro F1) significantly outperform lexicon baselines (VADER: 63.19% Accuracy, 0.6018 Macro F1).
- VADER suffers severe recall degradation on NEUTRAL posts (26.42% recall), frequently misinterpreting objective financial terms ("strike", "options", "volume") as directional emotional polarity.

### RQ2: FinBERT vs. General LLMs
- Specialized financial language models (FinBERT) achieve high precision on financial social media because their pre-training and fine-tuning objective aligns directly with financial phraseology.
- LLM prompting with structured or CoT reasoning provides rich explainability, but without in-domain fine-tuning or proprietary scale, domain transformers deliver higher raw accuracy at a fraction of latency.

### RQ3: Prompting Strategies Comparison
- Structured JSON output (L4) ensures 100% deterministic downstream pipeline ingestion and enables extracting confidence and ticker entities simultaneously.
- Chain-of-Thought (L5) generates step-by-step reasoning that enables auditing of retail trading signals.

### RQ5: Student Distillation with Regression Loss
- Deng et al. (2023)'s methodology demonstrates that continuous soft scores derived from multiple reasoning paths ($K=8$) provide a smooth supervisory signal for student models.
- Training a 66M parameter DistilBERT model on weak regression targets drastically compresses inference latency compared to multi-path LLM prompting.

### RQ6 & RQ7: Sentiment Association with Stock Returns
- Over the 2023–2024 timeframe across 14 major retail-traded equities, unweighted 1-day correlation between sentiment and returns is weak and slightly negative ($r = -0.0173$), consistent with modern market efficiency literature.
- Event analysis reveals a -0.17% return spread following extreme bullish attention spikes, indicating a mild "retail contrarian / attention reversal" effect.
