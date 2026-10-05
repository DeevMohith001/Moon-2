"""
Ticker and company detection for Reddit posts.
Identifies mentioned stock tickers, handles cash tags ($AAPL), resolves common names (Tesla -> TSLA),
and filters common financial/English acronym false positives.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from src.utils.common import get_logger

logger = get_logger(__name__)

# Common false positives: words that look like tickers (1-5 capital letters) but are English words or Reddit slang
STOP_TICKERS: Set[str] = {
    "A", "I", "AN", "AM", "AT", "AS", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "ALL", "ARE", "AND",
    "ANY", "BIG", "BUY", "CAN", "CAR", "CAT", "DOG", "DAY", "DUE", "FOR", "GET", "GOT", "HAS",
    "HAD", "HER", "HIM", "HIS", "HOW", "ITS", "LET", "LOT", "MAN", "MAY", "NEW", "NOT",
    "NOW", "OFF", "OLD", "ONE", "OUR", "OUT", "PAY", "PER", "PUT", "RUN", "SAY", "SEE",
    "SET", "SHE", "THE", "TOO", "TOP", "TRY", "TWO", "USE", "WAY", "WHO", "WHY", "WIN",
    "YES", "YET", "BEST", "CASH", "COOL", "COST", "DEAL", "DEEP", "DROP", "EASY", "EVEN",
    "EVER", "FALL", "FAST", "FEEL", "FILL", "FIND", "FINE", "FIRE", "FIVE", "FLAT", "FREE",
    "FULL", "FUND", "GAIN", "GIVE", "GLAD", "GOOD", "GROW", "HALF", "HARD", "HAVE", "HEAR",
    "HELD", "HELP", "HERE", "HIGH", "HOLD", "HOPE", "HUGE", "INTO", "JUST", "KEEP", "KNEW",
    "KNOW", "LAST", "LATE", "LEAD", "LEFT", "LESS", "LIFE", "LIKE", "LINE", "LIVE", "LONG",
    "LOOK", "LOSE", "LOSS", "LOST", "LOVE", "MADE", "MAKE", "MANY", "MARK", "MEAN", "MEET",
    "MIND", "MOON", "MORE", "MOST", "MOVE", "MUCH", "MUST", "NAME", "NEAR", "NEED", "NEXT",
    "NICE", "NONE", "OPEN", "OVER", "PART", "PAST", "PEAK", "PLAN", "PLAY", "POOR", "POST",
    "PUMP", "PULL", "PUSH", "RATE", "READ", "REAL", "REST", "RICH", "RIDE", "RISE", "RISK",
    "ROAD", "SAFE", "SAME", "SAVE", "SELL", "SEND", "SEEM", "SHOT", "SHOW", "SIDE", "SLOW",
    "SOME", "SOON", "STAY", "STOP", "SURE", "TAKE", "TALK", "TELL", "TERM", "TEST", "THAN",
    "THAT", "THEM", "THEN", "THEY", "THIS", "TIME", "TRUE", "TURN", "VERY", "VIEW", "WAIT",
    "WANT", "WEEK", "WELL", "WENT", "WHAT", "WHEN", "WILL", "WISH", "WITH", "WORD", "WORK",
    "YEAR", "ZERO",
    # Financial and Reddit Acronyms
    "DD", "RH", "WSB", "YOLO", "FOMO", "HODL", "ATH", "ATL", "BTFD", "TLDR", "IMO", "IMHO",
    "ROFL", "LMAO", "LOL", "WTF", "OMG", "RIP", "IRL", "ELI5", "POV", "PSA", "DCA", "NAV",
    "ETF", "IPO", "SPAC", "CEO", "CFO", "CTO", "COO", "SEC", "FED", "CPI", "GDP", "PCE",
    "EPS", "PE", "PEG", "ROI", "ROE", "ROA", "EBITDA", "EV", "IV", "RV", "DTE", "ITM",
    "OTM", "ATM", "FD", "LEAPS", "FUD", "MOASS", "SEC", "FINRA", "OTC", "NYSE", "NASDAQ",
    "FED", "FOMC", "JPOW", "BULL", "BEAR", "CALL", "PUT", "CALLS", "PUTS", "SHORT", "LONGS",
    "PUMP", "DUMP", "MEME", "STONK", "STONKS", "APE", "APES", "GUH", "LOSS", "GAIN", "EDIT",
}

# Major known stock tickers and company names
TICKER_NAME_MAP = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "GME": "GameStop",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet Google",
    "GOOG": "Alphabet Google",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "COIN": "Coinbase",
    "BA": "Boeing",
    "NFLX": "Netflix",
    "DIS": "Walt Disney",
    "PLTR": "Palantir",
    "BABA": "Alibaba",
    "AMC": "AMC Entertainment",
    "NIO": "NIO Inc",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "SOFI": "SoFi Technologies",
    "PYPL": "PayPal",
    "SQ": "Block / Square",
    "HOOD": "Robinhood",
    "BB": "BlackBerry",
    "CLOV": "Clover Health",
    "WISH": "ContextLogic Wish",
    "TLRY": "Tilray",
    "RIVN": "Rivian",
    "LCID": "Lucid Motors",
    "UBER": "Uber Technologies",
    "ABNB": "Airbnb",
    "SNOW": "Snowflake",
}

# Company name patterns mapping to tickers
COMPANY_NAME_PATTERNS = [
    (r"\b(apple|iphone|ipad|macbook)\b", "AAPL", "Apple"),
    (r"\b(tesla|elon|musk|cybertruck|model 3|model y|fsd)\b", "TSLA", "Tesla"),
    (r"\b(nvidia|jensen|geforce|rtx|cuda|h100|blackwell)\b", "NVDA", "Nvidia"),
    (r"\b(gamestop|game stop|cohen|roaring kitty|dfv)\b", "GME", "GameStop"),
    (r"\b(microsoft|windows|azure|copilot|bill gates)\b", "MSFT", "Microsoft"),
    (r"\b(amazon|bezos|aws|prime video)\b", "AMZN", "Amazon"),
    (r"\b(meta|facebook|zuckerberg|instagram|oculus)\b", "META", "Meta"),
    (r"\b(google|alphabet|sundar|gemini|deepmind|youtube)\b", "GOOGL", "Google"),
    (r"\b(amd|lisa su|ryzen|radeon)\b", "AMD", "AMD"),
    (r"\b(intel|pat gelsinger)\b", "INTC", "Intel"),
    (r"\b(coinbase|brian armstrong)\b", "COIN", "Coinbase"),
    (r"\b(boeing|737 max)\b", "BA", "Boeing"),
    (r"\b(netflix|squid game)\b", "NFLX", "Netflix"),
    (r"\b(disney|disney\+|bob iger|marvel|pixar)\b", "DIS", "Disney"),
    (r"\b(palantir|karp|alex karp)\b", "PLTR", "Palantir"),
    (r"\b(robinhood|vlad tenev)\b", "HOOD", "Robinhood"),
    (r"\b(rivian|r1t|r1s)\b", "RIVN", "Rivian"),
    (r"\b(lucid|lucid air)\b", "LCID", "Lucid Motors"),
    (r"\b(s&p\s*500|spy etf)\b", "SPY", "SPDR S&P 500 ETF"),
]


def extract_cashtags(text: str) -> List[str]:
    """Extract $TICKER occurrences from text."""
    if not text:
        return []
    matches = re.findall(r"\$([A-Za-z]{1,5})\b", text)
    valid = []
    for m in matches:
        ticker = m.upper()
        if ticker not in STOP_TICKERS or ticker in TICKER_NAME_MAP:
            valid.append(ticker)
    return valid


def extract_capital_tickers(text: str) -> List[str]:
    """Extract standalone uppercase ticker tokens (2-5 letters)."""
    if not text:
        return []
    tokens = re.findall(r"\b([A-Z]{2,5})\b", text)
    valid = []
    for token in tokens:
        if token in TICKER_NAME_MAP:
            valid.append(token)
        elif token not in STOP_TICKERS and token.isalpha():
            # Only accept unknown tickers if they are 3-4 letters and not stop words
            if len(token) in (3, 4):
                valid.append(token)
    return valid


def extract_company_names(text: str) -> List[Tuple[str, str]]:
    """Extract companies by name patterns, returning (ticker, company_name)."""
    if not text:
        return []
    found = []
    text_lower = text.lower()
    for pattern, ticker, comp_name in COMPANY_NAME_PATTERNS:
        if re.search(pattern, text_lower):
            found.append((ticker, comp_name))
    return found


def detect_tickers(text: str) -> Dict[str, any]:
    """
    Comprehensive ticker detection.
    Returns:
        {
            'primary_ticker': str or None,
            'primary_company': str or None,
            'all_tickers': list[str],
            'confidence': float,
            'detection_method': str
        }
    """
    if not text or not isinstance(text, str):
        return {
            "primary_ticker": None,
            "primary_company": None,
            "all_tickers": [],
            "confidence": 0.0,
            "detection_method": "none",
        }

    # 1. Cashtags (highest confidence)
    cashtags = extract_cashtags(text)
    if cashtags:
        # Count frequency
        counts = {}
        for c in cashtags:
            counts[c] = counts.get(c, 0) + 1
        primary = max(counts, key=counts.get)
        company = TICKER_NAME_MAP.get(primary, primary)
        return {
            "primary_ticker": primary,
            "primary_company": company,
            "all_tickers": list(set(cashtags)),
            "confidence": 0.95,
            "detection_method": "cashtag",
        }

    # 2. Known company name patterns
    companies = extract_company_names(text)
    if companies:
        primary_ticker, primary_comp = companies[0]
        all_t = list(set([t for t, _ in companies]))
        return {
            "primary_ticker": primary_ticker,
            "primary_company": primary_comp,
            "all_tickers": all_t,
            "confidence": 0.85,
            "detection_method": "company_name",
        }

    # 3. Capitalized ticker symbols
    capital_tickers = extract_capital_tickers(text)
    if capital_tickers:
        counts = {}
        for t in capital_tickers:
            counts[t] = counts.get(t, 0) + 1
        primary = max(counts, key=counts.get)
        company = TICKER_NAME_MAP.get(primary, primary)
        return {
            "primary_ticker": primary,
            "primary_company": company,
            "all_tickers": list(set(capital_tickers)),
            "confidence": 0.70 if primary in TICKER_NAME_MAP else 0.50,
            "detection_method": "capital_symbol",
        }

    return {
        "primary_ticker": None,
        "primary_company": None,
        "all_tickers": [],
        "confidence": 0.0,
        "detection_method": "none",
    }
