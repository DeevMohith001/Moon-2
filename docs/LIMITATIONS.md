# Limitations and Risk Analysis

## 1. Machine Learning & Hardware Constraints
- **Hardware Profile:** All transformer model inferences (`ProsusAI/finbert`, `yiyanghkust/finbert-tone`, `distilbert-base-uncased`) were conducted on CPU hardware. While sequence length capping (512 tokens) and batching optimize throughput, large-scale pretraining or full-scale distillation across millions of posts requires GPU acceleration.
- **Student Capacity:** Deng et al. (2023) used a 60M parameter Charformer student pre-trained on social media corpora. Our reproduction uses `distilbert-base-uncased` (66M parameters). While comparable in scale, DistilBERT uses WordPiece tokenization rather than character-level modeling, which may slightly increase vulnerability to novel retail misspellings and leetspeak.

---

## 2. Dataset & Sampling Biases
- **Retail Selection Bias:** Financial subreddits (especially `r/wallstreetbets`) skew heavily towards high-beta equities, options speculation, and mega-cap tech companies (NVDA, TSLA, GME, AAPL, MSFT), under-representing conservative equities, small-caps, and fixed income.
- **Survivorship & Moderation Censorship:** Moderated or auto-removed spam posts (`[removed]`, `[deleted]`) are filtered out during preprocessing. Highly controversial or manipulative posts may therefore be under-represented.

---

## 3. Financial Market & Econometric Limitations
- **Low Signal-to-Noise Ratio:** Daily equity returns are famously dominated by macro news, interest rate expectations, and institutional order flow. Retail sentiment accounts for only a minor fraction of short-term variance.
- **Non-Stationarity & Regime Shifts:** The relationship between Reddit sentiment and equity returns changed dramatically between the 2021 meme stock mania (GME/AMC short squeezes) and the 2023–2024 AI rally. Models calibrated on one market regime may not generalize to another.
- **Intraday vs. Daily Granularity:** In fast-moving markets, retail sentiment spikes may be priced in within minutes or hours. Daily forward return aggregation does not capture high-frequency intraday momentum or reversal effects.

---

## 4. LLM Prompting & Modeling Risks
- **Prompt Sensitivity:** Minor modifications to exemplar selection in few-shot prompts or system instructions can shift LLM classification boundaries.
- **Context Window & API Constraints:** Processing long discussion threads or multi-page Reddit self-posts requires truncation or chunking, potentially losing qualifying caveats located in concluding paragraphs.
- **Hallucination in Structured Output:** While JSON schema parsing filters invalid outputs, models occasionally hallucinate irrelevant ticker symbols when analyzing posts that discuss multiple competing firms.
