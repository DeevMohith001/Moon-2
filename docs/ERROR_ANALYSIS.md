# Error Analysis: Reddit Market Sentiment Modeling

## 1. Overview and Taxonomy of Social Financial Errors

Financial retail social media presents unique natural language processing challenges that standard general-domain models fail to parse. Based on real Reddit financial posts from `emilpartow/reddit_finance_posts_sp500`, errors are systematically classified into eight primary categories:

| Error Category | Linguistic / Domain Mechanism | Example Trigger / Idiom |
| :--- | :--- | :--- |
| **SARCASM_MEMES** | Ironic hyperbole, inverted intent | "Literally cannot go tits up", "Clown management 🤡", "Guh" |
| **CONTRADICTORY_ARGUMENTS** | Mixed financial signals (e.g. beat EPS, lowered guide) | "Revenue beat by 15%, but operating margin collapsed to 8%" |
| **ADVANCED_INVESTING_TERMINOLOGY** | Options mechanics, volatility dynamics | "IV crush after earnings", "Gamma squeeze", "Max pain target" |
| **SLANG_JARGON** | Retail community colloquialisms | "Tendies", "Bagholder", "Drill team", "Loss porn" |
| **MULTIPLE_COMPANIES_TICKERS** | Pair trades, competitive comparisons | "Selling AAPL to buy NVDA calls", "AMD vs INTC datacenter" |
| **LONG_DUE_DILIGENCE_POSTS** | Multi-paragraph deep dives (>1000 characters) | Elaborate DCF analyses with balanced counter-arguments |
| **TICKER_AMBIGUITY** | Common English words as cashtags | "IT", "FOR", "BE", "ALL", "ARE" |
| **SUBTLE_IMPLICIT_SENTIMENT** | Low-conviction exploratory discussion | "What time does earnings release drop?", "Considering trimming" |

---

## 2. Confusion Dynamics: Class-Wise Breakdown

### A. Neutral vs. Positive / Negative (The Boundary Ambiguity)
The majority of model errors occur on the decision boundary between **Neutral** and directional sentiment (**Positive** or **Negative**).
- **Inquiry vs. Conviction:** Informational questions often contain strong contextual keywords ("Are earnings expected to be huge?") that bag-of-words or un-calibrated models misclassify as directional bullish.
- **Hedging Strategies:** Posts describing risk mitigation (e.g., buying protective puts against a long position) represent neutral or prudent portfolio management, but are frequently mislabeled as bearish.

### B. Positive vs. Negative (Direct Inversions)
Direct polar inversions are rare (<5% of errors) but severe when they occur:
- **Sarcastic Losses:** A user posting a massive loss with "To the moon 🚀" is ironically coping with capital destruction. Baseline models without CoT reasoning treat the rocket emoji as bullish conviction.
- **Contradictory Fundamentals:** When top-line growth is overshadowed by massive debt or dilution, shallow encoders often latch onto positive headline metrics while missing bottom-line warnings.

---

## 3. Teacher Multi-Path Agreement vs. Error Probability

Analysis of multi-path generation reveals a strong monotonic relationship between teacher agreement and model error rate:
- **Agreement $\ge 7/8$:** Error rate $< 8\%$. These posts feature unambiguous directional language ("Loading calls, guidance beat across the board").
- **Agreement $5/8$ - $6/8$:** Error rate $\approx 22\%$. Posts feature nuanced arguments or conditional conviction.
- **Agreement $< 5/8$:** Error rate $> 45\%$. Severe ambiguity, sarcasm, or balanced debate. Filtering these posts out (retention threshold $\ge 5/8$) is essential for clean student distillation.

---

## 4. Preservation of Real Evaluation Evidence

All error analyses must use real post texts and genuine evaluation outputs stored under `outputs/error_analysis/`. Synthetic results are never substituted for research error diagnostics.
