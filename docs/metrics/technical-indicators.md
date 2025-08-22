# Technical Indicators Documentation

This document details all technical indicators and market analysis metrics implemented in OptiTrade.

## Overview

OptiTrade implements various technical indicators for market analysis, trend identification, and trading signal generation. These indicators are sourced from multiple data providers and calculated using historical price data.

## Moving Averages

### Simple Moving Averages (SMA)

**SMA20 (20-Day Simple Moving Average)**
- **Formula**: Average of closing prices over 20 trading days
- **Use**: Short-term trend identification
- **Source**: `finvizfinance/constants.py` (line 48)

**SMA50 (50-Day Simple Moving Average)**
- **Formula**: Average of closing prices over 50 trading days
- **Use**: Medium-term trend identification
- **Source**: `finvizfinance/constants.py` (line 49)

**SMA200 (200-Day Simple Moving Average)**
- **Formula**: Average of closing prices over 200 trading days
- **Use**: Long-term trend identification
- **Source**: `finvizfinance/constants.py` (line 50)

### Moving Average Applications

Moving averages are used for:
- **Trend Direction**: Price above/below MA indicates trend
- **Support/Resistance**: MAs act as dynamic support/resistance levels
- **Crossover Signals**: Price or MA crossovers generate trading signals
- **Trend Strength**: Distance from MA indicates trend strength

## Momentum Indicators

### Relative Strength Index (RSI)

**RSI (Relative Strength Index)**
- **Formula**: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
- **Range**: 0 to 100
- **Interpretation**: 
  - RSI > 70: Potentially overbought
  - RSI < 30: Potentially oversold
- **Source**: `finvizfinance/constants.py` (line 55)

### Price Performance Indicators

**from Open**
- **Description**: Percentage change from opening price
- **Use**: Intraday momentum assessment
- **Source**: `finvizfinance/constants.py` (line 56)

**Gap**
- **Description**: Percentage gap between previous close and current open
- **Use**: Identifies price gaps and potential reversal points
- **Source**: `finvizfinance/constants.py` (line 57)

## Volatility Indicators

### Average True Range (ATR)

**ATR (Average True Range)**
- **Formula**: Average of True Range over specified period
- **True Range**: Max of (High-Low, |High-PrevClose|, |Low-PrevClose|)
- **Use**: Measures market volatility
- **Source**: `finvizfinance/constants.py` (line 45)

### Volatility Calculations

**Implementation in fetch_nasdaq.py**:
```python
# Volatility Calculation
prices = hist["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()
vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()
annualized_vol = vol_series.iloc[-1] * np.sqrt(252)
```

**Volatility Windows**:
- **Volatility W**: Weekly volatility measure
- **Volatility M**: Monthly volatility measure
- **Source**: `finvizfinance/constants.py` (lines 46-47)

**Parameters**:
- **Rolling Window**: 30 days (configurable)
- **Annualization Factor**: √252 (trading days per year)
- **Location**: `scripts/fetch_nasdaq.py` (lines 149-154)

## Price Level Indicators

### High/Low Analysis

**52W High (52-Week High)**
- **Description**: Highest price in the last 52 weeks
- **Use**: Resistance level identification
- **Source**: `finvizfinance/constants.py` (line 53)

**52W Low (52-Week Low)**
- **Description**: Lowest price in the last 52 weeks
- **Use**: Support level identification
- **Source**: `finvizfinance/constants.py` (line 54)

**50D High/Low (50-Day High/Low)**
- **Description**: Highest/lowest price in the last 50 days
- **Use**: Medium-term support/resistance levels
- **Source**: `finvizfinance/constants.py` (lines 51-52)

## Volume Indicators

### Volume Analysis

**Avg Volume (Average Volume)**
- **Description**: Average trading volume over specified period
- **Use**: Liquidity assessment and volume trend analysis
- **Source**: `finvizfinance/constants.py` (line 59)

**Rel Volume (Relative Volume)**
- **Description**: Current volume relative to average volume
- **Formula**: Current Volume / Average Volume
- **Use**: Identifies unusual trading activity
- **Source**: `finvizfinance/constants.py` (line 60)

**Volume**
- **Description**: Current trading volume
- **Use**: Confirms price movements and trend strength
- **Source**: `finvizfinance/constants.py` (line 63)

## Performance Metrics

### Time-Based Performance

OptiTrade tracks performance across multiple timeframes:

**Short-term Performance**:
- **Perf Week**: Weekly performance percentage
- **Perf Month**: Monthly performance percentage

**Medium-term Performance**:
- **Perf Quart**: Quarterly performance percentage
- **Perf Half**: Half-year performance percentage

**Long-term Performance**:
- **Perf Year**: Annual performance percentage
- **Perf YTD**: Year-to-date performance percentage

**Source**: `finvizfinance/constants.py` (lines 38-43)

### Performance Categories

The system includes detailed performance filtering options:

```python
"Performance": {
    "prefix": "ta_perf",
    "option": {
        "Today Up": "dup",
        "Today Down": "ddown",
        "Today -15%": "d15u",
        "Today -10%": "d10u",
        "Today -5%": "d5u",
        "Today +5%": "d5o",
        "Today +10%": "d10o",
        "Today +15%": "d15o",
        "Week -30%": "1w30u",
        "Week -20%": "1w20u",
        "Week -10%": "1w10u",
        "Week Down": "1wdown",
        "Week Up": "1wup",
        "Week +10%": "1w10o",
        "Week +20%": "1w20o",
        "Week +30%": "1w30o"
        # ... additional performance ranges
    }
}
```

## Risk Indicators

### Beta

**Beta**
- **Description**: Measures stock's volatility relative to market
- **Interpretation**:
  - Beta = 1: Moves with market
  - Beta > 1: More volatile than market
  - Beta < 1: Less volatile than market
- **Source**: `finvizfinance/constants.py` (line 44)

## Market Data Integration

### Real-time Price Data

**Current Price Fetching**:
```python
async def fetch_current_price():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://www.tradingview.com/symbols/NASDAQ-IXIC/")
        # Parse TradingView data for real-time prices
```

**Location**: `app/routers/indices.py` (lines 12-24)

### NASDAQ Index Calculations

**Daily Change Calculation**:
```python
daily_change = round(current - previous_close, 2)
pct_change = round((daily_change / previous_close) * 100, 2)
```

**Intraday Data Processing**:
```python
intraday = [
    {
        "time": row["Datetime"].isoformat(),
        "open": round(row["Open"], 2),
        "close": round(row["Close"], 2),
        "high": round(row["High"], 2),
        "low": round(row["Low"], 2),
        "volume": int(row["Volume"]),
    }
    for _, row in intraday_df.iterrows()
]
```

## Implementation Architecture

### Data Sources

1. **Yahoo Finance**: Primary source for OHLCV data
2. **Finviz**: Technical indicators and screening data
3. **TradingView**: Real-time price feeds

### Calculation Engine

**Location**: `scripts/fetch_nasdaq.py`

**Key Features**:
- Multi-threaded data processing
- Retry mechanisms for data reliability
- Comprehensive error handling
- Real-time data validation

**Configuration**:
```python
MAX_WORKERS = 32  # Concurrent processing threads
ROLLING_WINDOW = 30  # Days for volatility calculation
RETRY_LIMIT = 3  # Maximum retry attempts
```

### Data Validation

**Historical Data Requirements**:
```python
if (
    hist.empty
    or "Close" not in hist
    or hist["Close"].dropna().shape[0] < 40
):
    raise ValueError("Insufficient historical data for volatility calculation.")
```

**Equity Validation**:
```python
if not info or info.get("quoteType") != "EQUITY":
    raise ValueError("Not an equity or no info available.")
```

## Usage in Trading Strategies

Technical indicators support various trading strategies:

1. **Trend Following**: Using moving averages and momentum indicators
2. **Mean Reversion**: Using RSI and volatility measures
3. **Breakout Trading**: Using support/resistance levels
4. **Volume Analysis**: Using volume indicators for confirmation

## Performance Optimization

- **Caching**: Technical indicators are cached to reduce computation
- **Batch Processing**: Multiple indicators calculated in single data fetch
- **Parallel Processing**: Multi-threaded calculation for large datasets
- **Data Compression**: Efficient storage of historical indicator values
