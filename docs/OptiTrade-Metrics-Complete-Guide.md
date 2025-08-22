# OptiTrade Metrics - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Portfolio Metrics](#portfolio-metrics)
3. [Financial Ratios](#financial-ratios)
4. [Technical Indicators](#technical-indicators)
5. [Risk Metrics](#risk-metrics)
6. [Performance Metrics](#performance-metrics)
7. [Market Data Metrics](#market-data-metrics)
8. [Implementation Details](#implementation-details)
9. [Calculation Examples](#calculation-examples)

## Overview

OptiTrade implements a comprehensive suite of financial metrics and calculations for portfolio management, risk assessment, and market analysis. All calculations use decimal precision to ensure accuracy and are updated in real-time with market data.

**Key Features (ACTUALLY IMPLEMENTED):**
- Real-time portfolio valuation ✅
- Portfolio performance tracking ✅
- Historical volatility calculation ✅
- Dividend frequency analysis ✅
- Concentration risk assessment ✅
- Market data integration (OHLCV) ✅
- NASDAQ index performance ✅

**External Data Integration (NOT CALCULATED BY US):**
- Financial ratios (from Finviz)
- Technical indicators (from Finviz)
- Company fundamental data (from Yahoo Finance)

## Portfolio Metrics

### Core Portfolio Calculations

**Net Worth**
- **Formula**: `Net Worth = Cash Balance + Portfolio Value`
- **Implementation**: `user_balance.net_worth = (cash_balance + portfolio_value).quantize(Decimal("0.01"))`
- **Location**: `scripts/portfolio_management.py`

**Portfolio Value**
- **Formula**: `Portfolio Value = Σ(Quantity × Current Price)` for all holdings
- **Updates**: Real-time with market price changes
- **Precision**: 2 decimal places

**Average Price (Cost Basis)**
- **Formula**: `Average Price = (Previous Total Cost + New Purchase Cost) / Total Quantity`
- **Implementation**:
```python
portfolio_item.average_price = (
    (portfolio_quantity * portfolio_avg_price + total_cost) / new_quantity
).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**Total Invested**
- **Formula**: `Total Invested = Quantity × Average Price`
- **Purpose**: Tracks total capital deployed per position

**Current Value per Position**
- **Formula**: `Current Value = Quantity × Current Market Price`
- **Updates**: Real-time market price updates

### Transaction Impact Calculations

**Buy Transaction Impact**:
```python
# Cash reduction
new_cash_balance = (current_balance - total_cost).quantize(Decimal("0.01"))

# Portfolio value increase
new_portfolio_value = (current_portfolio + quantity * price).quantize(Decimal("0.01"))

# Net worth update
user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(Decimal("0.01"))
```

**Sell Transaction Impact**:
```python
# Cash increase
new_cash_balance = (current_balance + total_sale).quantize(Decimal("0.01"))

# Portfolio value decrease
new_portfolio_value = (current_portfolio - quantity * price).quantize(Decimal("0.01"))

# Position adjustment for partial sales
if new_quantity > 0:
    portfolio_item.total_invested = (new_quantity * average_price).quantize(Decimal("0.01"))
else:
    # Complete position closure
    await db.delete(portfolio_item)
```

## Financial Ratios (EXTERNAL DATA - NOT CALCULATED BY US)

**Important Note**: OptiTrade does NOT calculate financial ratios. These are retrieved from external sources (Finviz) for display purposes only.

### Available Through Finviz Integration (65+ ratios)

**Valuation Ratios** (from Finviz):
- P/E, Forward P/E, PEG, P/S, P/B, P/C, P/FCF

**Profitability Ratios** (from Finviz):
- ROA, ROE, ROI, Gross Margin, Operating Margin, Profit Margin

**Liquidity Ratios** (from Finviz):
- Current Ratio, Quick Ratio

**Leverage Ratios** (from Finviz):
- Long-term Debt/Equity, Total Debt/Equity

**Dividend Metrics** (from Finviz):
- Dividend yield, Payout ratio

**Technical Indicators** (from Finviz):
- SMA20, SMA50, SMA200, RSI, Beta, ATR

**Performance Metrics** (from Finviz):
- Weekly, monthly, quarterly, yearly performance

### What We Actually Calculate

**Only Dividend Frequency Analysis**:
```python
# ACTUAL implementation in scripts/fetch_nasdaq.py
def get_dividend_frequency(dividends):
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

**Data Source**: Yahoo Finance dividend history
**Location**: `scripts/fetch_nasdaq.py` (lines 101-117)

## Technical Indicators (ACTUALLY IMPLEMENTED)

### Volatility Calculation (ONLY ONE WE ACTUALLY CALCULATE)

**Historical Volatility**
- **Formula**: 30-day rolling standard deviation of log returns, annualized
- **Implementation**:
```python
# ACTUAL implementation in scripts/fetch_nasdaq.py (lines 149-154)
prices = hist["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()
vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()
annualized_vol = vol_series.iloc[-1] * np.sqrt(252)
```
- **Parameters**: Rolling window = 30 days, Annualization factor = √252
- **Location**: `scripts/fetch_nasdaq.py`
- **Output**: `30d_realized_volatility_annualized` in stock data

### Dividend Frequency Analysis (ACTUALLY IMPLEMENTED)

**Dividend Payment Frequency**:
```python
# ACTUAL implementation in scripts/fetch_nasdaq.py (lines 101-117)
def get_dividend_frequency(dividends):
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

### Market Data Processing (ACTUALLY IMPLEMENTED)

**NASDAQ Index Performance**:
```python
# ACTUAL implementation in app/routers/indices.py
daily_change = round(current - previous_close, 2)
pct_change = round((daily_change / previous_close) * 100, 2)
```

**Price Data Processing**:
- **OHLCV Data**: Open, High, Low, Close, Volume from Yahoo Finance
- **Market Cap**: From company info
- **Dividend Data**: Last dividend date and amount
- **Company Info**: Industry, sector, website data

### External Data Integration (NOT CALCULATED BY US)

**Note**: The following are available through Finviz integration but NOT calculated by OptiTrade:
- Moving averages (SMA20, SMA50, SMA200)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Financial ratios (P/E, P/B, ROE, etc.)
- Beta coefficient
- Volume indicators

These are retrieved from external sources, not computed by our algorithms.

## Risk Metrics (ACTUALLY IMPLEMENTED)

### Volatility Risk (CALCULATED BY US)

**Historical Volatility**
- **Calculation**: 30-day rolling standard deviation of log returns, annualized
- **Implementation**:
```python
# ACTUAL code in scripts/fetch_nasdaq.py (lines 149-154)
prices = hist["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()
vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()
annualized_vol = vol_series.iloc[-1] * np.sqrt(252)
```
- **Parameters**: Rolling window = 30 days, Annualization factor = √252
- **Location**: `scripts/fetch_nasdaq.py`

### Concentration Risk (CALCULATED BY US)

**Position Sizing Analysis**:
```python
# Portfolio concentration calculation
position_weight = (quantity * current_price) / total_portfolio_value

# Risk assessment
if position_weight > 0.30:  # 30% threshold
    print("⚠️ High concentration risk detected")
```

**Implementation**: Real-time calculation during portfolio operations
**Location**: Portfolio management functions

### Portfolio Risk Assessment (FRAMEWORK EXISTS)

**Risk Monitoring Framework**:
```python
# Example framework (not fully implemented)
async def assess_portfolio_risk(user_id, db):
    portfolio = await get_user_portfolio(user_id, db)

    # Calculate portfolio metrics
    total_value = sum(position.current_value for position in portfolio)

    # Concentration risk
    max_position = max(position.current_value for position in portfolio)
    concentration_ratio = max_position / total_value

    return {
        "concentration_risk": concentration_ratio,
        "volatility_risk": portfolio_volatility,
    }
```

### External Risk Data (NOT CALCULATED BY US)

**Available from Finviz but NOT calculated by OptiTrade**:
- Beta coefficient (market risk)
- Financial leverage ratios
- Liquidity ratios
- Credit risk indicators

## Performance Metrics

### Return Calculations

**Basic Returns**
- **Simple Return**: (Current Value - Initial Investment) ÷ Initial Investment
- **Percentage Return**: ((Current Value ÷ Initial Investment) - 1) × 100

**Time-Weighted Returns**
- **Perf Week**: Weekly performance percentage
- **Perf Month**: Monthly performance percentage
- **Perf Quart**: Quarterly performance percentage
- **Perf Half**: Half-year performance percentage
- **Perf Year**: Annual performance percentage
- **Perf YTD**: Year-to-date performance percentage

### Portfolio Performance

**Total Portfolio Return**:
```python
def calculate_portfolio_return(portfolio_items):
    total_current_value = sum(item.current_value for item in portfolio_items)
    total_invested = sum(item.total_invested for item in portfolio_items)
    
    if total_invested == 0:
        return 0
    
    return (total_current_value - total_invested) / total_invested
```

### Risk-Adjusted Performance

**Sharpe Ratio**
- **Formula**: (Portfolio Return - Risk-free Rate) ÷ Portfolio Standard Deviation
- **Implementation**:
```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)
```

### Historical Performance Tracking

**Portfolio History**
- Daily snapshots stored in `portfolio_history` table
- Location: `scripts/portfolio_history.py`
- Prevents duplicate entries for same date
- Supports trend analysis and performance attribution

## Market Data Metrics

### Price Data (OHLCV)

**Basic Price Metrics**
- **Open, High, Low, Close, Volume**: Standard OHLCV data
- **Source**: Yahoo Finance via yfinance
- **Frequency**: Daily and intraday (1-hour intervals)

### Index Performance

**NASDAQ Composite Analysis**:
```python
async def nasdaq_summary():
    current = await fetch_current_price()
    previous_close = ticker.info.get("previousClose", 0)
    
    daily_change = round(current - previous_close, 2)
    pct_change = round((daily_change / previous_close) * 100, 2)
    
    return {
        "current": current,
        "previous_close": previous_close,
        "daily_change": daily_change,
        "percent_change": pct_change,
    }
```

### Volume Analysis

**Volume Metrics**
- **Current Volume**: Real-time trading volume
- **Average Volume**: Historical average over specified period
- **Relative Volume**: Current volume relative to average

**Volume-Price Relationship**:
```python
def calculate_vwap(prices, volumes):
    """Volume-Weighted Average Price"""
    return (prices * volumes).sum() / volumes.sum()
```

## Implementation Details

### Data Sources

**Primary Sources**
1. **Yahoo Finance**: OHLCV data, company information
2. **Finviz**: Financial ratios, screening data
3. **TradingView**: Real-time price feeds

### Precision and Rounding

**Decimal Precision**
- All monetary calculations use `Decimal` type
- Rounding method: `ROUND_HALF_UP`
- Currency precision: 2 decimal places
- Ratio precision: 4 decimal places

### Error Handling

**Safe Calculations**:
```python
def safe_calculate_return(current_value, invested_value):
    try:
        if invested_value == 0:
            return 0.0
        return (current_value - invested_value) / invested_value
    except (ZeroDivisionError, TypeError):
        return 0.0
```

### Performance Optimization

**Caching and Efficiency**
- Timed cache for data retrieval (`@timed_cache`)
- Multi-threaded processing (32 workers)
- Batch calculations for multiple positions
- Real-time price feed service

### Data Validation

**Quality Checks**:
```python
# Historical data validation
if (hist.empty or "Close" not in hist or hist["Close"].dropna().shape[0] < 40):
    raise ValueError("Insufficient historical data")

# Equity validation
if not info or info.get("quoteType") != "EQUITY":
    raise ValueError("Not an equity or no info available")
```

## Calculation Examples

### Example 1: Portfolio Value Calculation

**Scenario**: User has AAPL (150 shares at $155) and GOOGL (25 shares at $2,800)

```python
# Individual positions
aapl_value = 150 * Decimal('155.00')  # $23,250.00
googl_value = 25 * Decimal('2800.00')  # $70,000.00

# Total portfolio value
portfolio_value = aapl_value + googl_value  # $93,250.00

# With cash balance
cash_balance = Decimal('5000.00')
net_worth = cash_balance + portfolio_value  # $98,250.00
```

### Example 2: Average Price Calculation

**Scenario**: Buy 100 shares at $150, then 50 shares at $160

```python
# First purchase
quantity_1, price_1 = 100, Decimal('150.00')
total_cost_1 = quantity_1 * price_1  # $15,000

# Second purchase
quantity_2, price_2 = 50, Decimal('160.00')
total_cost_2 = quantity_2 * price_2  # $8,000

# New average price
new_quantity = quantity_1 + quantity_2  # 150
new_average = (total_cost_1 + total_cost_2) / new_quantity  # $153.33
```

### Example 3: Volatility Calculation (ACTUALLY IMPLEMENTED)

```python
# ACTUAL 30-day volatility calculation from scripts/fetch_nasdaq.py
prices = hist["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()
vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()  # ROLLING_WINDOW = 30
annualized_vol = vol_series.iloc[-1] * np.sqrt(252)

# Result stored as:
result["30d_realized_volatility_annualized"] = round(annualized_vol, 6)
```

### Example 4: Dividend Frequency Analysis (ACTUALLY IMPLEMENTED)

```python
# ACTUAL dividend frequency calculation from scripts/fetch_nasdaq.py
def get_dividend_frequency(dividends):
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

# Usage in data processing:
dividends = stock.dividends
payment_frequency = get_dividend_frequency(dividends)
```

### Example 5: Portfolio Concentration Risk (CALCULATED BY US)

```python
# Concentration risk assessment
max_position_value = 70000.00
total_portfolio_value = 93250.00
concentration_ratio = max_position_value / total_portfolio_value  # 75.07%

if concentration_ratio > 0.30:  # 30% threshold
    print("⚠️ High concentration risk detected")
```

## Configuration Parameters

**Risk Thresholds**:
```python
RISK_THRESHOLDS = {
    "MAX_POSITION_WEIGHT": 0.20,        # 20% max position size
    "MAX_SECTOR_WEIGHT": 0.30,          # 30% max sector allocation
    "MAX_PORTFOLIO_VOLATILITY": 0.25,   # 25% max portfolio volatility
    "MIN_LIQUIDITY_VOLUME": 100000,     # Minimum daily volume
}
```

**Calculation Parameters**:
```python
MAX_WORKERS = 32           # Concurrent processing threads
ROLLING_WINDOW = 30        # Days for volatility calculation
RETRY_LIMIT = 3           # Maximum retry attempts
CACHE_TTL_SECONDS = 300   # Cache time-to-live
```

This comprehensive guide covers all metrics implemented in OptiTrade, providing both theoretical understanding and practical implementation details for portfolio management, risk assessment, and market analysis.
