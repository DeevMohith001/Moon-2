# Reddit Financial Sentiment Dataset Documentation

## Overview

This dataset supports the reproduction and extension of the research paper:
**"What do LLMs Know about Financial Markets?"** (Deng et al., 2023, arXiv:2212.11311).

The dataset consists of financial social media posts from Reddit communities focused on equities and trading (such as `r/wallstreetbets`, `r/stocks`, `r/investing`, and `r/StockMarket`), capturing both retail investor sentiment, financial terminology, sarcasm, and stock discussions.

---

## Dataset Splits

| Split | Description | Ratio | Purpose |
|-------|-------------|-------|---------|
| **Train** | Primary training split | 70% | Baseline ML training, distillation student training |
| **Validation** | Hyperparameter tuning | 15% | Model selection, early stopping, prompt tuning |
| **Test** | Holdout evaluation | 15% | Unseen evaluation of all models, error analysis |

*Note: In addition to stratified splits for classification benchmarks, chronological splitting is supported to eliminate lookahead bias when testing market return correlations.*

---

## Schema & Attributes

| Column | Type | Description |
|--------|------|-------------|
| `post_id` | String | Unique post identifier |
| `timestamp` | Datetime (UTC) | Time of post creation (normalized to ISO 8601 UTC) |
| `subreddit` | String | Source community (`wallstreetbets`, `stocks`, `investing`, `StockMarket`) |
| `text` | String | Cleaned text combining title and selftext |
| `ticker` | String | Primary identified equity ticker (e.g. `NVDA`, `AAPL`, `TSLA`, `SPY`) |
| `company` | String | Canonical corporate name |
| `all_tickers` | List[String] | All recognized tickers present in the post |
| `ticker_confidence`| Float | Confidence score of ticker extraction (0.0 to 1.0) |
| `detection_method` | String | Method (`cashtag`, `company_name`, `capital_symbol`) |
| `sentiment_label` | String | Ground truth or annotated label (`BULLISH`, `BEARISH`, `NEUTRAL`) |
| `sentiment_score` | Float | Normalized scalar sentiment (`+1.0`, `-1.0`, `0.0`) |
| `score` | Integer | Reddit post upvote score |
| `num_comments` | Integer | Total comment count |
| `char_length` | Integer | Total character count of cleaned text |
| `word_count` | Integer | Total word count |

---

## Preprocessing Pipeline

The raw Reddit text undergoes specialized cleaning tailored to social financial discussions:

1. **HTML/Markdown Decoding**:
   - Entities like `&amp;`, `&lt;` are unescaped.
   - Markdown bold, italics, code blocks, quote blocks are stripped.
2. **URL Handling**:
   - Web links are removed while preserving surrounding context.
3. **Financial Emoji Preservation**:
   - High-signal sentiment emojis (e.g., 🚀, 💎🙌, 📉, 🐂, 🐻, 🤡) are mapped to semantic tokens (`rocket_bullish`, `diamond_hands`, `bear_market`, `clown_sarcastic`) rather than dropped.
4. **Length and Quality Filtering**:
   - Deleted/removed posts (`[deleted]`, `[removed]`) are discarded.
   - Posts with fewer than 15 characters of content are excluded.
5. **Deduplication**:
   - Duplicate post texts across subreddits or reposts are removed.
6. **Entity Recognition & Ticker Extraction**:
   - Distinguishes cash tags (`$TSLA`), known company names, and capital symbols while filtering out over 100 common English false positives (e.g., `A`, `IT`, `FOR`, `ON`, `BUY`, `HOLD`, `CEO`, `FED`, `DD`).
