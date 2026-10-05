# Methodology Documentation

## LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models
**Reference:** Reproduction and Extension of Deng et al. (2023), *"What do LLMs Know about Financial Markets?"* (arXiv:2212.11311)

---

## 1. Research Objectives & Conceptual Framework

Financial social media (e.g., Reddit communities `r/wallstreetbets`, `r/stocks`, `r/investing`, `r/StockMarket`) exhibits unique linguistic characteristics including extreme sarcasm, memes, retail trading jargon ("calls", "puts", "diamond hands 💎🙌", "to the moon 🚀", "IV crush"), and mixed sentiment.

This project addresses two foundational research questions inspired by Deng et al. (2023):
1. **NLP & Distillation:** Can large language models (LLMs) accurately interpret retail financial sentiment, and can their knowledge be distilled into a lightweight student model via continuous regression loss on agreement ratios?
2. **Financial Market Integration:** Does aggregated Reddit market sentiment contain predictive signal for subsequent equity returns, or is retail attention primarily noise?

---

## 2. Data Acquisition, Cleaning & Ticker Extraction

### 2.1 Preprocessing Pipeline
Raw Reddit submissions undergo a domain-specific normalization pipeline:
1. **HTML & Markdown Cleaning:** Decodes entities (`&amp;`, `&lt;`) and strips markdown formatting.
2. **URL Sanitization:** Strips external hyperlinks while retaining surrounding discussion text.
3. **Sentiment Emoji Preservation:** High-signal sentiment emojis (🚀, 💎🙌, 📉, 🐂, 🐻, 🤡) are mapped to semantic tokens (`EMOJI_ROCKET_BULLISH`, `EMOJI_DIAMOND_HANDS`, `EMOJI_BEARISH_TREND`) rather than dropped.
4. **Length and Quality Filtering:** Posts under 10 characters or marked `[deleted]` / `[removed]` are filtered out.
5. **Stratified and Chronological Splitting:** 
   - 70% Train (760 samples)
   - 15% Validation (163 samples)
   - 15% Holdout Test (163 samples)

### 2.2 Ticker Detection Hierarchy
To prevent false-positive ticker identification (such as detecting common English words like `A`, `FOR`, `ON`, `IT`, `BUY`, `HOLD`, `CEO`, `FED`):
- **Priority 1: Cashtag Matching (`$AAPL`):** High confidence (0.95), regex matching `\$([A-Z]{1,5})`.
- **Priority 2: Corporate Entity Lookup:** Maps known company names ("NVIDIA", "Microsoft", "Apple") to canonical symbols.
- **Priority 3: Capitalized Ticker Filtering:** Matches standalone uppercase symbols against a curated financial whitelist and over 100 negative stop-words.

---

## 3. Modeling Taxonomy

| Model Class | Experiment ID | Architecture / Base | Objective / Loss |
|---|---|---|---|
| **Lexicon** | B1 | VADER + Financial Slang Lexicon | Compound Polarity Thresholding ([-0.05, 0.05]) |
| **Classical ML** | B2 | TF-IDF (1-2 grams) + Logistic Regression | Cross-Entropy with Validation C Tuning |
| **Classical ML** | B3 | TF-IDF (1-2 grams) + Linear SVM | Hinge Loss with Validation C Tuning |
| **Domain Transformer** | B4 | `ProsusAI/finbert` | Pre-trained Financial Sequence Classification |
| **Domain Transformer** | B5 | `yiyanghkust/finbert-tone` | Pre-trained Financial Tone Sequence Classification |
| **General LLM** | L1 | Zero-Shot Direct Labeling | Prompting (`prompt_a_zeroshot.txt`) |
| **General LLM** | L2 | 6-Shot Exemplar (2 Bull, 2 Bear, 2 Neutral) | Prompting (`prompt_b_fewshot.txt`) |
| **General LLM** | L3 | Financial Market Context & Definitions | Prompting (`prompt_c_financial_context.txt`) |
| **General LLM** | L4 | Structured JSON Schema Extraction | Prompting (`prompt_d_structured.txt`) |
| **General LLM** | L5 | Chain-of-Thought (CoT) 4-Step Reasoning | Prompting (`prompt_e_cot.txt`) |
| **Distilled Student** | R2 | `distilbert-base-uncased` | **MSE Regression Loss** on Teacher Agreement Scores |

---

## 4. Distillation Methodology (Reproducing Deng et al., 2023)

### 4.1 Teacher Weak Supervision
1. For each Reddit post, the teacher LLM generates $K = 8$ distinct reasoning paths at temperature $\tau = 0.5$ using the 4-step CoT prompt.
2. For each reasoning path $k \in \{1, \dots, 8\}$, sentiment is parsed into scalar labels: $y_k \in \{+1.0 \text{ (Bullish)}, 0.0 \text{ (Neutral)}, -1.0 \text{ (Bearish)}\}$.
3. **Consistency Filtering:** The sample is retained only if the majority class achieves at least 5 out of 8 path consensus ($\ge 62.5\%$ agreement). Inconsistent samples are discarded.
4. **Continuous Soft Target:** The soft sentiment score $s_i$ is computed as the mean of the reasoning paths:
   $$s_i = \frac{1}{K} \sum_{k=1}^K y_{i, k} \in [-1.0, +1.0]$$

### 4.2 Student Model Training
- **Student Architecture:** `distilbert-base-uncased` with a linear regression head ($num\_labels = 1$).
- **Loss Function:** Mean Squared Error (MSE) regression loss:
   $$\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^N (\hat{s}_i - s_i)^2$$
- **Inference Thresholding:** The continuous prediction $\hat{s}$ is mapped to 3 classes using a tuned validation threshold $\theta$:
   $$\hat{y} = \begin{cases} \text{BULLISH} & \text{if } \hat{s} > \theta \\ \text{BEARISH} & \text{if } \hat{s} < -\theta \\ \text{NEUTRAL} & \text{otherwise} \end{cases}$$

---

## 5. Financial Econometrics & Return Prediction

### 5.1 Temporal Alignment
Posts are timestamped in UTC. US equity markets operate from 09:30 to 16:00 US/Eastern.
- Posts created **after 16:00 ET** cannot be traded until day $t+1$, and are assigned to trading session $t+1$.
- Posts on weekends or holidays map to the next trading calendar date.
- This eliminates lookahead bias.

### 5.2 Daily Sentiment Aggregation
Daily ticker sentiment is aggregated across multiple metrics:
1. **Unweighted Mean Sentiment:** $\bar{S}_{i, t} = \frac{1}{M} \sum_{m=1}^M s_{i, t, m}$
2. **Bull-Bear Sentiment Ratio:** $\frac{N_{bull} - N_{bear}}{N_{bull} + N_{bear} + N_{neutral}}$
3. **Engagement-Weighted Sentiment:** Weighted by $\log(1 + \text{score} + \text{num\_comments})$

### 5.3 Return Horizons and Predictive Modeling
Forward returns are computed using adjusted close prices:
$$R_{t, t+h} = \frac{\text{AdjClose}_{t+h} - \text{AdjClose}_t}{\text{AdjClose}_t}, \quad h \in \{1, 3, 5\} \text{ trading days}$$
We compare:
- **Model A (Market Baseline):** Lagged 1-day momentum $\to$ Direction($R_{t, t+1}$)
- **Model B (Sentiment Baseline):** Daily sentiment features $\to$ Direction($R_{t, t+1}$)
- **Model C (Combined Model):** Market momentum + Daily sentiment $\to$ Direction($R_{t, t+1}$)
