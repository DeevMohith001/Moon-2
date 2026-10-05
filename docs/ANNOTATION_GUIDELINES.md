# Annotation Guidelines: Reddit Financial Market Sentiment

## 1. Class Definitions

### BULLISH (+1.0)
The author expresses positive expectations regarding the target company's share price, business fundamentals, earnings, or market valuation:
- **Explicit Actions:** Buying shares, buying call options, averaging up, opening long positions.
- **Fundamental Catalysts:** Beating revenue/EPS expectations, increasing forward guidance, expanding gross margins, key product adoption, new contract wins.
- **Social & Slang Indicators:** "To the moon 🚀", "breakout", "undervalued", "buying the dip", "diamond hands 💎🙌", "short squeeze candidate".

### BEARISH (-1.0)
The author expresses negative expectations regarding the target company's future stock performance:
- **Explicit Actions:** Buying put options, short selling, selling off shares, cutting losses ("bagholding").
- **Fundamental Catalysts:** Missing earnings estimates, lowering guidance, CEO departures under investigation, debt default risk, dilution through secondary offerings.
- **Social & Slang Indicators:** "Drilling", "tanking", "drill team", "rug pull", "overvalued bubble", "clown management 🤡", "loss porn".

### NEUTRAL (0.0)
The author presents balanced, informational, or objective discussion without clear directional conviction:
- **Informational Inquiries:** Asking for earnings release times, discussing SEC filings (10-K, 10-Q) without opinion, portfolio rebalancing discussions.
- **Balanced Debates:** Analyzing both bullish growth prospects and bearish valuation multiples equally.
- **Macroeconomic Commentary:** Broad macroeconomic observations (e.g., Fed interest rate decisions, CPI prints) without taking a stance on a specific stock.

---

## 2. Challenging Edge Cases & Disambiguation Rules

### 2.1 Sarcasm and Irony
Retail forums frequently use sarcastic hyperbole:
- *"Literally cannot go tits up"* $\to$ Usually **BEARISH / High-Risk Warning** (used ironically before a collapse).
- *"Thank you for your donation, bears"* $\to$ **BULLISH** (boasting about an upward price surge).
- *Clown emoji 🤡 after earnings report* $\to$ **BEARISH** (mocking leadership incompetence).

### 2.2 Options Trading Context
- **Writing / Selling Covered Calls:** Neutral to mildly bullish on the stock, but capped upside. Classify as **NEUTRAL** unless directional stock bias is stated.
- **Hedging:** *"Buying puts to protect my shares"* $\to$ **NEUTRAL** (overall long risk mitigation).

### 2.3 Mixed Time Horizons
When short-term pain is paired with long-term optimism (*"Short term looking rough due to supply chains, but 2-year horizon is ultra bullish"*):
- Classify by the **predominant investment recommendation** of the post. If balanced, label as **NEUTRAL**.
