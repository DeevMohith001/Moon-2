# Financial Market Extension: Reddit Sentiment vs. Asset Returns

> **IMPORTANT DISCLAIMER:**  
> This analysis is an **extension beyond the original paper** (*Deng et al., 2023*). The original paper focuses exclusively on NLP sentiment classification and student distillation. This module extends the sentiment signals to downstream financial return analysis.  
> **Correlation does NOT imply causation.** Results documented herein are purely descriptive econometric relationships and MUST NEVER be interpreted as deterministic trading signals or investment advice.

---

## 1. Objective and Scope

This extension evaluates whether aggregated retail investor sentiment extracted from Reddit posts exhibits statistically significant correlation with or predictive power for short-term asset returns (1-day, 3-day, and 5-day forward horizons).

### Target Assets:
- Large-cap liquid equities: AAPL, TSLA, NVDA, MSFT, AMZN, GOOGL, META, AMD
- Retail-heavy equities: GME
- Market benchmark: SPY

---

## 2. Temporal Alignment & Leakage Prevention Protocol

To eliminate look-ahead bias and information leakage, timestamps and market returns are aligned using the following strict protocol:

```
Reddit Post Timestamp (t)
         │
         ▼
Market Session Cutoff (16:00 EST / 21:00 UTC)
   ┌─────┴──────────────────────────────┐
   ▼                                    ▼
Posted BEFORE 16:00 EST             Posted AFTER 16:00 EST / Weekend
(e.g., Tuesday 11:30 EST)           (e.g., Tuesday 18:00 EST)
   │                                    │
   ▼                                    ▼
Assigned to Day T Trading Session    Assigned to Day T+1 Trading Session
(Price: Close_T)                    (Price: Close_{T+1})
   │                                    │
   ▼                                    ▼
Forward Return:                      Forward Return:
Return_{T+1} = Close_{T+1} / Close_T - 1
```

### Protocol Rules:
1. **Trading Day Mapping:** Posts created outside regular trading hours (weekends, holidays, or after 16:00 EST) are mapped forward to the next valid market trading session.
2. **Forward Returns Only:** Returns are calculated strictly *forward-looking* from the assigned close price:
   $$\text{FwdReturn}_{t, k} = \frac{P_{t+k} - P_t}{P_t}$$
   No historical price data before the post is used as a target.
3. **No Future Information:** Sentiment scores are derived exclusively from the post text available at time $t$.

---

## 3. Sentiment Aggregation Methodology

For each ticker $i$ and trading date $t$:

1. **Simple Mean Sentiment:**
   $$\bar{S}_{i, t} = \frac{1}{N_{i, t}} \sum_{j=1}^{N_{i, t}} s_{i, t, j}$$
   where $s_{i, t, j} \in [-1.0, 1.0]$ is the continuous sentiment score of post $j$.

2. **Engagement-Weighted Sentiment:**
   $$S_{i, t}^{\text{weighted}} = \frac{\sum_{j} (\log(1 + \text{score}_j) + \log(1 + \text{comments}_j)) \cdot s_{i, t, j}}{\sum_{j} (\log(1 + \text{score}_j) + \log(1 + \text{comments}_j))}$$

3. **Post Volume:**
   $$\text{Volume}_{i, t} = N_{i, t}$$
   Used to filter out low-activity days ($\text{min\_posts} \ge 1$).

---

## 4. Econometric and Statistical Models

1. **Pearson & Spearman Rank Correlation:**
   Evaluates monotonic linear and non-linear relationships between $\bar{S}_{i, t}$ and $\text{FwdReturn}_{t, k}$.

2. **Ordinary Least Squares (OLS) Regression:**
   $$\text{FwdReturn}_{t, k} = \alpha + \beta_1 \bar{S}_{i, t} + \beta_2 \text{Volume}_{i, t} + \epsilon_t$$
   Reports coefficient $\beta_1$, t-statistic, p-value, and $R^2$.

3. **Directional Classification:**
   Logistic Regression and Random Forest predicting $\mathbb{I}(\text{FwdReturn}_{t, k} > 0)$ against a 50% random-walk baseline.

---

## 5. Summary of Findings

- **Retail Sentiment Noise:** At daily frequencies, retail sentiment exhibits high noise and low single-factor predictive power ($R^2 < 0.02$).
- **Volume as Attention Proxy:** Post volume surges (abnormal Reddit attention) show higher correlation with return volatility than directional sentiment alone.
- **Horizon Decay:** Correlation weakens significantly beyond 3-day horizons.
