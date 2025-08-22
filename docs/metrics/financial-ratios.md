# Financial Ratios Documentation

This document details all financial ratios and fundamental analysis metrics available through the Finviz integration in OptiTrade.

## Overview

OptiTrade integrates with Finviz to provide comprehensive financial ratios and fundamental analysis data. These metrics are available through the screener modules and provide deep insights into company financial health and valuation.

## Valuation Ratios

### Price-to-Earnings Ratios

**P/E (Price-to-Earnings)**
- **Description**: Current stock price divided by earnings per share
- **Use**: Measures how much investors are willing to pay per dollar of earnings
- **Source**: `finvizfinance/constants.py` (line 3)

**Fwd P/E (Forward P/E)**
- **Description**: Stock price divided by forward-looking earnings estimates
- **Use**: Valuation based on expected future earnings
- **Source**: `finvizfinance/constants.py` (line 4)

**PEG (Price/Earnings to Growth)**
- **Description**: P/E ratio divided by earnings growth rate
- **Use**: Accounts for growth when evaluating P/E ratios
- **Source**: `finvizfinance/constants.py` (line 5)

### Price-to-Book and Related Ratios

**P/S (Price-to-Sales)**
- **Description**: Market cap divided by total revenue
- **Use**: Valuation metric especially useful for companies with no earnings
- **Source**: `finvizfinance/constants.py` (line 6)

**P/B (Price-to-Book)**
- **Description**: Stock price divided by book value per share
- **Use**: Compares market value to accounting book value
- **Source**: `finvizfinance/constants.py` (line 7)

**P/C (Price-to-Cash)**
- **Description**: Market cap divided by cash per share
- **Use**: Measures premium paid for company's cash position
- **Source**: `finvizfinance/constants.py` (line 8)

**P/FCF (Price-to-Free Cash Flow)**
- **Description**: Market cap divided by free cash flow
- **Use**: Values company based on cash generation ability
- **Source**: `finvizfinance/constants.py` (line 9)

## Profitability Ratios

### Return Metrics

**ROA (Return on Assets)**
- **Formula**: Net Income / Total Assets
- **Description**: Measures how efficiently company uses assets to generate profit
- **Source**: `finvizfinance/constants.py` (line 28)

**ROE (Return on Equity)**
- **Formula**: Net Income / Shareholders' Equity
- **Description**: Measures return generated on shareholders' equity
- **Source**: `finvizfinance/constants.py` (line 29)

**ROI (Return on Investment)**
- **Formula**: (Gain - Cost) / Cost
- **Description**: Measures efficiency of investment
- **Source**: `finvizfinance/constants.py` (line 30)

### Margin Analysis

**Gross M (Gross Margin)**
- **Formula**: (Revenue - COGS) / Revenue
- **Description**: Percentage of revenue retained after direct costs
- **Source**: `finvizfinance/constants.py` (line 35)

**Oper M (Operating Margin)**
- **Formula**: Operating Income / Revenue
- **Description**: Percentage of revenue retained after operating expenses
- **Source**: `finvizfinance/constants.py` (line 36)

**Profit M (Profit Margin)**
- **Formula**: Net Income / Revenue
- **Description**: Percentage of revenue converted to profit
- **Source**: `finvizfinance/constants.py` (line 37)

## Liquidity Ratios

**Curr R (Current Ratio)**
- **Formula**: Current Assets / Current Liabilities
- **Description**: Measures ability to pay short-term obligations
- **Source**: `finvizfinance/constants.py` (line 31)

**Quick R (Quick Ratio)**
- **Formula**: (Current Assets - Inventory) / Current Liabilities
- **Description**: More stringent liquidity measure excluding inventory
- **Source**: `finvizfinance/constants.py` (line 32)

## Leverage Ratios

**LTDebt/Eq (Long-term Debt to Equity)**
- **Formula**: Long-term Debt / Total Equity
- **Description**: Measures long-term financial leverage
- **Source**: `finvizfinance/constants.py` (line 33)

**Debt/Eq (Total Debt to Equity)**
- **Formula**: Total Debt / Total Equity
- **Description**: Measures overall financial leverage
- **Source**: `finvizfinance/constants.py` (line 34)

## Dividend Metrics

**Dividend**
- **Description**: Annual dividend per share
- **Use**: Income generation assessment
- **Source**: `finvizfinance/constants.py` (line 10)

**Payout Ratio**
- **Formula**: Dividends per Share / Earnings per Share
- **Description**: Percentage of earnings paid as dividends
- **Source**: `finvizfinance/constants.py` (line 11)

### Dividend Analysis Implementation

**Dividend Frequency Calculation**:
```python
def get_dividend_frequency(dividends):
    """Estimates dividend frequency based on payment dates."""
    if dividends.empty or len(dividends) < 2:
        return None
    
    dividend_dates = dividends.index.to_series().diff().dt.days.dropna()
    avg_days_between = dividend_dates.mean()
    
    if 80 <= avg_days_between <= 100:
        return "Quarterly"
    elif 170 <= avg_days_between <= 200:
        return "Semi-Annually"
    elif 350 <= avg_days_between <= 380:
        return "Annually"
    else:
        return "Irregular"
```

**Location**: `scripts/fetch_nasdaq.py` (lines 101-117)

## Earnings Metrics

**EPS (Earnings Per Share)**
- **Description**: Net income divided by outstanding shares
- **Source**: `finvizfinance/constants.py` (line 12)

**EPS Growth Metrics**:
- **EPS this Y**: Current year EPS growth
- **EPS next Y**: Next year EPS growth estimate
- **EPS past 5Y**: Historical 5-year EPS growth
- **EPS next 5Y**: Projected 5-year EPS growth
- **EPS Q/Q**: Quarter-over-quarter EPS growth

**Sales Growth Metrics**:
- **Sales past 5Y**: Historical 5-year sales growth
- **Sales Q/Q**: Quarter-over-quarter sales growth

## Share Structure Metrics

**Outstanding**
- **Description**: Total shares outstanding
- **Source**: `finvizfinance/constants.py` (line 20)

**Float**
- **Description**: Shares available for public trading
- **Source**: `finvizfinance/constants.py` (line 21)

**Ownership Metrics**:
- **Insider Own**: Percentage owned by insiders
- **Insider Trans**: Recent insider transaction activity
- **Inst Own**: Institutional ownership percentage
- **Inst Trans**: Institutional transaction activity

**Short Interest**:
- **Float Short**: Percentage of float sold short
- **Short Ratio**: Days to cover short positions

## Implementation Details

### Data Access

Financial ratios are accessed through Finviz screener modules:

```python
from finvizfinance.screener.financial import Financial
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.overview import Overview
```

### Data Processing

All numerical columns are automatically processed:
```python
NUMBER_COL = [
    "Market Cap", "P/E", "Fwd P/E", "PEG", "P/S", "P/B", "P/C", "P/FCF",
    "Dividend", "Payout Ratio", "EPS", "ROA", "ROE", "ROI",
    "Curr R", "Quick R", "LTDebt/Eq", "Debt/Eq",
    "Gross M", "Oper M", "Profit M"
    # ... and more
]
```

### Screening Capabilities

The platform supports filtering stocks based on financial ratios:
- Custom ratio ranges
- Multiple criteria combinations
- Performance-based screening
- Fundamental analysis filters

## Usage in OptiTrade

These financial ratios are used for:
1. **Stock Screening**: Filter stocks based on fundamental criteria
2. **Investment Analysis**: Evaluate potential investments
3. **Portfolio Analysis**: Assess current holdings
4. **Risk Assessment**: Identify financially stable companies
5. **Valuation Analysis**: Determine fair value estimates

## Data Quality and Updates

- **Source**: Finviz provides regularly updated financial data
- **Frequency**: Data updates align with company reporting cycles
- **Validation**: Automatic data type conversion and validation
- **Error Handling**: Graceful handling of missing or invalid data
