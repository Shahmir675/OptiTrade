# Market Data Metrics Documentation

This document details all market data metrics, calculations, and analysis tools implemented in OptiTrade for comprehensive market analysis.

## Overview

OptiTrade integrates multiple market data sources to provide comprehensive market analysis metrics. These metrics cover price analysis, volume analysis, market indices, and real-time market conditions.

## Price Metrics

### Basic Price Data (OHLCV)

**Open, High, Low, Close, Volume (OHLCV)**
- **Source**: Yahoo Finance via yfinance
- **Frequency**: Daily, intraday (1-hour intervals)
- **Implementation**: `scripts/fetch_nasdaq.py`, `app/routers/indices.py`

**Price Data Structure**:
```python
{
    "time": "2024-01-15T09:30:00",
    "open": 150.25,
    "high": 152.80,
    "low": 149.90,
    "close": 151.75,
    "volume": 1250000
}
```

### Price Change Calculations

**Daily Price Change**:
```python
daily_change = round(current - previous_close, 2)
pct_change = round((daily_change / previous_close) * 100, 2)
```

**Implementation Location**: `app/routers/indices.py` (lines 35-36)

**Intraday Price Movements**:
```python
# From opening price
from_open_pct = (current_price - open_price) / open_price * 100

# Gap analysis
gap_pct = (open_price - previous_close) / previous_close * 100
```

### Price Level Analysis

**Support and Resistance Levels**:
- **52W High**: 52-week highest price
- **52W Low**: 52-week lowest price  
- **50D High**: 50-day highest price
- **50D Low**: 50-day lowest price

**Source**: `finvizfinance/constants.py` (lines 51-54)

**Target Price Analysis**:
- **Target Price**: Analyst consensus target price
- **Current vs Target**: Price deviation from target
- **Source**: `finvizfinance/constants.py` (line 64)

## Volume Metrics

### Volume Analysis

**Average Volume Calculation**:
```python
def calculate_average_volume(volume_data, period=30):
    """Calculate average volume over specified period"""
    return volume_data.rolling(window=period).mean().iloc[-1]
```

**Relative Volume**:
```python
def calculate_relative_volume(current_volume, average_volume):
    """Calculate relative volume ratio"""
    return current_volume / average_volume if average_volume > 0 else 0
```

**Volume Metrics Available**:
- **Volume**: Current trading volume
- **Avg Volume**: Average volume over period
- **Rel Volume**: Current volume / Average volume

**Source**: `finvizfinance/constants.py` (lines 59-63)

### Volume-Price Analysis

**Volume-Weighted Average Price (VWAP)**:
```python
def calculate_vwap(prices, volumes):
    """Calculate Volume-Weighted Average Price"""
    return (prices * volumes).sum() / volumes.sum()
```

**Volume Trend Analysis**:
```python
def analyze_volume_trend(volume_data, price_data):
    """Analyze volume trends with price movements"""
    volume_ma = volume_data.rolling(window=20).mean()
    price_change = price_data.pct_change()
    
    # Volume confirmation analysis
    up_days = price_change > 0
    down_days = price_change < 0
    
    avg_volume_up = volume_data[up_days].mean()
    avg_volume_down = volume_data[down_days].mean()
    
    return {
        'volume_trend': 'bullish' if avg_volume_up > avg_volume_down else 'bearish',
        'volume_ratio': avg_volume_up / avg_volume_down if avg_volume_down > 0 else float('inf')
    }
```

## Market Index Metrics

### NASDAQ Composite Analysis

**Real-time Index Data**:
```python
async def fetch_current_price():
    """Fetch real-time NASDAQ price from TradingView"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://www.tradingview.com/symbols/NASDAQ-IXIC/")
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all("script", {"type": "application/prs.init-data+json"})
        
        raw_json = script_tags[3].string or script_tags[3].text
        data = json.loads(raw_json)
        outer_key = next(iter(data))
        
        return round(float(data[outer_key]["data"]["symbol"]["trade"]["price"]), 2)
```

**Location**: `app/routers/indices.py` (lines 12-24)

**Index Performance Metrics**:
```python
{
    "current": 15234.56,
    "previous_close": 15180.23,
    "daily_change": 54.33,
    "percent_change": 0.36
}
```

### Intraday Index Analysis

**Hourly Index Data**:
```python
async def nasdaq_intraday():
    """Get NASDAQ intraday hourly data"""
    ticker = yf.Ticker("^IXIC")
    intraday_df = await asyncio.to_thread(
        lambda: ticker.history(period="1d", interval="1h")
    )
    
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
    
    return {"intraday": intraday}
```

**Location**: `app/routers/indices.py` (lines 49-75)

## Market Screening Metrics

### Finviz Integration

**Market Screening Categories**:
1. **Overview**: Basic market data and ratios
2. **Valuation**: P/E, P/B, P/S ratios
3. **Financial**: ROE, ROA, debt ratios
4. **Ownership**: Institutional and insider ownership
5. **Performance**: Price performance across timeframes
6. **Technical**: Technical indicators and signals

**Available Screeners**:
```python
from finvizfinance.screener import (
    Overview, Valuation, Financial, Ownership, 
    Performance, Technical, Custom
)
```

### Performance Screening

**Performance Categories**:
```python
PERFORMANCE_FILTERS = {
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
}
```

**Source**: `finvizfinance/constants.py` (lines 1396-1430)

## Real-time Data Processing

### Price Feed Service

**Continuous Price Updates**:
```python
async def start_price_fetching_task():
    """Start background task for continuous price updates"""
    tickers = get_all_tickers()
    if tickers:
        asyncio.create_task(external_fetch_prices(tickers))
```

**Location**: `app/services/price_feed_service.py` (lines 14-17)

**Price Data Access**:
```python
def get_current_price_data():
    """Get current price data from memory"""
    return external_price_data
```

### Data Validation and Quality

**Historical Data Validation**:
```python
# Data quality checks
if (
    hist.empty
    or "Close" not in hist
    or hist["Close"].dropna().shape[0] < 40
):
    raise ValueError("Insufficient historical data for volatility calculation.")

# Equity validation
if not info or info.get("quoteType") != "EQUITY":
    raise ValueError("Not an equity or no info available.")
```

**Location**: `scripts/fetch_nasdaq.py` (lines 137-146)

## Market Data Storage

### Stock Data Caching

**Timed Cache Implementation**:
```python
@timed_cache(ttl_seconds=CACHE_TTL_SECONDS)
def download_stock_data_sync(
    stock_symbol: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "max",
    retries: int = 2,
):
    # Implementation with retry logic and caching
```

**Location**: `app/services/data_service.py` (lines 105-133)

### Data Persistence

**Stock Data JSON Storage**:
```python
# Configuration for data storage
SAVE_PATH = "../app/static/stocks.json"

# Data structure for storage
stock_data = {
    "symbol": ticker,
    "company_name": info.get("longName", "N/A"),
    "current_price": float(latest_ohlcv["Close"]),
    "market_cap": info.get("marketCap", 0),
    "volume": int(latest_ohlcv["Volume"]),
    "volatility": float(annualized_vol),
    "dividend_yield": info.get("dividendYield", 0),
    "pe_ratio": info.get("trailingPE", 0),
    "beta": info.get("beta", 0),
    "52_week_high": info.get("fiftyTwoWeekHigh", 0),
    "52_week_low": info.get("fiftyTwoWeekLow", 0)
}
```

## Market Analysis Tools

### Technical Analysis Integration

**Moving Average Analysis**:
- **SMA20**: 20-day simple moving average
- **SMA50**: 50-day simple moving average  
- **SMA200**: 200-day simple moving average

**Momentum Indicators**:
- **RSI**: Relative Strength Index
- **Beta**: Market correlation coefficient
- **ATR**: Average True Range

### Fundamental Analysis

**Financial Health Metrics**:
- **Current Ratio**: Liquidity measure
- **Debt/Equity**: Leverage analysis
- **ROE/ROA**: Profitability metrics
- **P/E Ratios**: Valuation metrics

## Error Handling and Reliability

### Retry Mechanisms

**Data Fetching with Retries**:
```python
for attempt in range(RETRY_LIMIT):
    try:
        # Fetch data
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period="60d", interval="1d")
        info = stock.info
        
        # Validate data
        if hist.empty or "Close" not in hist:
            raise ValueError("Insufficient data")
            
        return process_data(hist, info)
        
    except Exception as e:
        last_error = e
        attempt += 1
        if attempt < RETRY_LIMIT:
            time.sleep(0.5)
```

### Failure Handling

**Consecutive Failure Monitoring**:
```python
with failure_lock:
    consecutive_failures["count"] += 1
    if consecutive_failures["count"] >= FAILURE_THRESHOLD:
        restart_script()
```

**Configuration**:
- **RETRY_LIMIT**: 3 attempts per ticker
- **FAILURE_THRESHOLD**: 20 consecutive failures
- **MAX_WORKERS**: 32 concurrent threads

## Performance Optimization

### Concurrent Processing

**Multi-threaded Data Fetching**:
```python
MAX_WORKERS = 32  # I/O bound optimization
ROLLING_WINDOW = 30  # Volatility calculation window

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_ticker_data, ticker, volume_data.get(ticker, 0)): ticker
        for ticker in tickers
    }
```

### Memory Management

**Efficient Data Structures**:
- Pandas DataFrames for time series analysis
- JSON serialization for API responses
- In-memory caching for frequently accessed data
- Batch processing for multiple tickers

## API Integration

### External Data Sources

**Primary Sources**:
1. **Yahoo Finance**: OHLCV data, company information
2. **Finviz**: Financial ratios, screening data
3. **TradingView**: Real-time price feeds

**Secondary Sources**:
- **Alpha Vantage**: Alternative price data
- **IEX Cloud**: Market data backup
- **Quandl**: Economic indicators

### Rate Limiting

**API Call Management**:
```python
# Session management for rate limiting
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0...'})

# Delay between requests
time.sleep(0.1)  # 100ms delay between API calls
```
