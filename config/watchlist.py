"""
Default NSE watchlist with sector classification.
Used for sector exposure caps (Section 21) and LLM concentration checks (Section 10.1).
"""
from typing import Dict

# WATCHLIST_WITH_SECTORS: Dict[str, str] = {
#     "RELIANCE":    "Energy",
#     "TCS":         "IT",
#     "HDFCBANK":    "Banking",
#     "INFY":        "IT",
#     "ICICIBANK":   "Banking",
#     "HINDUNILVR":  "FMCG",
#     "ITC":         "FMCG",
#     "SBIN":        "Banking",
#     "BHARTIARTL":  "Telecom",
#     "KOTAKBANK":   "Banking",
#     "LT":          "Infrastructure",
#     "AXISBANK":    "Banking",
#     "ASIANPAINT":  "Paints",
#     "MARUTI":      "Automobile",
#     "TITAN":       "Consumer Discretionary",
# }

WATCHLIST_WITH_SECTORS = {
    "SBIN": "Banking",
    "COALINDIA": "Energy",
    "SUNPHARMA": "Pharma",
    "M&M": "Auto",
    "EICHERMOT": "Auto",
    "NTPC": "Energy",
    "POWERGRID": "Energy",
}

# NSE index membership — used for slippage tier (Section 20.1)
NIFTY50_STOCKS = {
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
}

NIFTY_NEXT50_STOCKS: set = set()     # populate as watchlist grows
