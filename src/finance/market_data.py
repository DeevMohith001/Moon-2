"""
Market Data Acquisition and Trading Calendar Alignment.

Uses yfinance with local disk caching and aligns Reddit post timestamps
to the appropriate NYSE/NASDAQ trading day.
"""

from typing import List, Optional
import pandas as pd
from src.finance.prices import fetch_historical_prices
from src.finance.alignment import align_posts_to_trading_days

__all__ = [
    "fetch_historical_prices",
    "align_posts_to_trading_days",
]
