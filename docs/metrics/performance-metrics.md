# Performance Metrics Documentation

This document details all performance measurement and analysis metrics implemented in OptiTrade for tracking investment and portfolio performance.

## Overview

OptiTrade provides comprehensive performance metrics to evaluate investment success across multiple timeframes and dimensions. These metrics help users understand returns, compare performance, and make informed investment decisions.

## Return Calculations

### Basic Return Metrics

**Simple Return**
- **Formula**: (Current Value - Initial Investment) / Initial Investment
- **Implementation**:
```python
simple_return = (current_value - total_invested) / total_invested
```

**Percentage Return**
- **Formula**: ((Current Value / Initial Investment) - 1) × 100
- **Use**: Standard performance measurement

### Time-Weighted Returns

**Daily Returns**
```python
daily_return = (today_close - yesterday_close) / yesterday_close
```

**Period Returns**:
- **Perf Week**: Weekly performance percentage
- **Perf Month**: Monthly performance percentage  
- **Perf Quart**: Quarterly performance percentage
- **Perf Half**: Half-year performance percentage
- **Perf Year**: Annual performance percentage
- **Perf YTD**: Year-to-date performance percentage

**Source**: `finvizfinance/constants.py` (lines 38-43)

## Portfolio Performance Metrics

### Portfolio-Level Returns

**Total Portfolio Return**:
```python
def calculate_portfolio_return(portfolio_items):
    total_current_value = sum(item.current_value for item in portfolio_items)
    total_invested = sum(item.total_invested for item in portfolio_items)
    
    if total_invested == 0:
        return 0
    
    return (total_current_value - total_invested) / total_invested
```

**Weighted Average Return**:
```python
def calculate_weighted_return(portfolio_items):
    total_value = sum(item.current_value for item in portfolio_items)
    
    weighted_return = 0
    for item in portfolio_items:
        weight = item.current_value / total_value
        item_return = (item.current_value - item.total_invested) / item.total_invested
        weighted_return += weight * item_return
    
    return weighted_return
```

### Position-Level Performance

**Individual Position Returns**:
```python
# Unrealized gain/loss per position
unrealized_pnl = current_value - total_invested
unrealized_return = unrealized_pnl / total_invested

# Realized gain/loss (from transactions)
realized_pnl = sell_proceeds - (quantity_sold * average_price)
```

**Implementation Location**: `scripts/portfolio_management.py`

## Historical Performance Tracking

### Portfolio History Analysis

**Daily Portfolio Snapshots**:
```python
def update_portfolio_history(conn, unique_ids):
    """Store daily portfolio snapshots for performance tracking"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM portfolio
            WHERE user_id = ANY(%s)
        """, (unique_ids,))
        
        records = cur.fetchall()
        snapshot_date = "NOW()"
        values = [(*record, snapshot_date) for record in records]
        
        cur.executemany("""
            INSERT INTO portfolio_history
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, values)
```

**Location**: `scripts/portfolio_history.py` (lines 78-104)

### Performance Trend Analysis

**Time Series Performance**:
```python
def calculate_performance_trend(portfolio_history):
    """Calculate performance trends from historical data"""
    dates = [record.snapshot_date for record in portfolio_history]
    values = [record.current_value for record in portfolio_history]
    
    # Calculate daily returns
    daily_returns = []
    for i in range(1, len(values)):
        daily_return = (values[i] - values[i-1]) / values[i-1]
        daily_returns.append(daily_return)
    
    return {
        'daily_returns': daily_returns,
        'cumulative_return': (values[-1] - values[0]) / values[0],
        'volatility': np.std(daily_returns) * np.sqrt(252)
    }
```

## Market Performance Comparison

### Index Performance Tracking

**NASDAQ Index Performance**:
```python
async def nasdaq_summary():
    """Calculate NASDAQ daily performance metrics"""
    current = await fetch_current_price()
    ticker = yf.Ticker("^IXIC")
    previous_close = await asyncio.to_thread(
        lambda: round(ticker.info.get("previousClose", 0), 2)
    )
    
    daily_change = round(current - previous_close, 2)
    pct_change = round((daily_change / previous_close) * 100, 2)
    
    return {
        "current": current,
        "previous_close": previous_close,
        "daily_change": daily_change,
        "percent_change": pct_change,
    }
```

**Location**: `app/routers/indices.py` (lines 28-46)

### Benchmark Comparison

**Relative Performance**:
```python
def calculate_relative_performance(portfolio_return, benchmark_return):
    """Calculate performance relative to benchmark"""
    return portfolio_return - benchmark_return

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """Calculate tracking error vs benchmark"""
    excess_returns = portfolio_returns - benchmark_returns
    return np.std(excess_returns) * np.sqrt(252)
```

## Risk-Adjusted Performance

### Sharpe Ratio

**Formula**: (Portfolio Return - Risk-free Rate) / Portfolio Standard Deviation

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio for risk-adjusted performance"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)
```

### Information Ratio

**Formula**: (Portfolio Return - Benchmark Return) / Tracking Error

```python
def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculate information ratio"""
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    return excess_returns.mean() / tracking_error * np.sqrt(252)
```

## Transaction Performance Analysis

### Trade Analysis

**Individual Trade Performance**:
```python
def analyze_trade_performance(transactions):
    """Analyze performance of individual trades"""
    buy_transactions = [t for t in transactions if t.transaction_type == 'buy']
    sell_transactions = [t for t in transactions if t.transaction_type == 'sell']
    
    trade_results = []
    for sell in sell_transactions:
        # Find corresponding buy transactions (FIFO)
        remaining_quantity = sell.quantity
        total_cost_basis = 0
        
        for buy in buy_transactions:
            if buy.symbol == sell.symbol and remaining_quantity > 0:
                quantity_used = min(remaining_quantity, buy.quantity)
                total_cost_basis += quantity_used * buy.price_per_share
                remaining_quantity -= quantity_used
        
        if remaining_quantity == 0:
            profit_loss = (sell.quantity * sell.price_per_share) - total_cost_basis
            trade_results.append({
                'symbol': sell.symbol,
                'quantity': sell.quantity,
                'profit_loss': profit_loss,
                'return_pct': profit_loss / total_cost_basis
            })
    
    return trade_results
```

### Win/Loss Analysis

**Trade Statistics**:
```python
def calculate_trade_statistics(trade_results):
    """Calculate win/loss statistics"""
    winning_trades = [t for t in trade_results if t['profit_loss'] > 0]
    losing_trades = [t for t in trade_results if t['profit_loss'] < 0]
    
    return {
        'total_trades': len(trade_results),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trade_results) if trade_results else 0,
        'avg_win': np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0,
        'avg_loss': np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0,
        'profit_factor': abs(sum(t['profit_loss'] for t in winning_trades) / 
                           sum(t['profit_loss'] for t in losing_trades)) if losing_trades else float('inf')
    }
```

## Performance Attribution

### Sector Performance

**Sector-wise Returns**:
```python
def calculate_sector_performance(portfolio_items, sector_mapping):
    """Calculate performance by sector"""
    sector_performance = {}
    
    for item in portfolio_items:
        sector = sector_mapping.get(item.symbol, 'Unknown')
        if sector not in sector_performance:
            sector_performance[sector] = {
                'total_invested': 0,
                'current_value': 0,
                'positions': []
            }
        
        sector_performance[sector]['total_invested'] += item.total_invested
        sector_performance[sector]['current_value'] += item.current_value
        sector_performance[sector]['positions'].append(item)
    
    # Calculate sector returns
    for sector in sector_performance:
        data = sector_performance[sector]
        data['return'] = (data['current_value'] - data['total_invested']) / data['total_invested']
        data['weight'] = data['current_value'] / sum(s['current_value'] for s in sector_performance.values())
    
    return sector_performance
```

### Asset Allocation Performance

**Performance by Asset Class**:
```python
def analyze_allocation_performance(portfolio_items, allocation_targets):
    """Analyze performance vs target allocation"""
    total_value = sum(item.current_value for item in portfolio_items)
    
    actual_allocation = {}
    for item in portfolio_items:
        asset_class = get_asset_class(item.symbol)
        if asset_class not in actual_allocation:
            actual_allocation[asset_class] = 0
        actual_allocation[asset_class] += item.current_value / total_value
    
    allocation_drift = {}
    for asset_class, target in allocation_targets.items():
        actual = actual_allocation.get(asset_class, 0)
        allocation_drift[asset_class] = actual - target
    
    return allocation_drift
```

## Performance Reporting

### Performance Dashboard Metrics

**Key Performance Indicators**:
1. Total return (absolute and percentage)
2. Annualized return
3. Risk-adjusted returns (Sharpe ratio)
4. Maximum drawdown
5. Win/loss ratio
6. Benchmark relative performance

### Performance Visualization Data

**Chart Data Preparation**:
```python
def prepare_performance_chart_data(portfolio_history):
    """Prepare data for performance charts"""
    dates = []
    values = []
    returns = []
    
    for i, record in enumerate(portfolio_history):
        dates.append(record.snapshot_date.isoformat())
        values.append(float(record.current_value))
        
        if i > 0:
            daily_return = (record.current_value - portfolio_history[i-1].current_value) / portfolio_history[i-1].current_value
            returns.append(daily_return)
    
    return {
        'dates': dates,
        'values': values,
        'returns': returns[1:] if len(returns) > 1 else [],
        'cumulative_return': (values[-1] - values[0]) / values[0] if len(values) > 1 else 0
    }
```

## Performance Optimization

### Calculation Efficiency

**Batch Performance Calculations**:
```python
async def calculate_portfolio_performance_batch(user_ids, db):
    """Calculate performance for multiple users efficiently"""
    # Fetch all portfolio data in single query
    portfolios = await db.execute(
        select(Portfolio).where(Portfolio.user_id.in_(user_ids))
    )
    
    # Group by user and calculate performance
    user_performance = {}
    for portfolio_item in portfolios.scalars():
        user_id = portfolio_item.user_id
        if user_id not in user_performance:
            user_performance[user_id] = []
        user_performance[user_id].append(portfolio_item)
    
    # Calculate performance metrics for each user
    results = {}
    for user_id, items in user_performance.items():
        results[user_id] = calculate_portfolio_return(items)
    
    return results
```

### Caching Strategy

**Performance Data Caching**:
```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=1000)
def get_cached_performance(user_id, date_str):
    """Cache daily performance calculations"""
    # Implementation for cached performance retrieval
    pass

def invalidate_performance_cache(user_id):
    """Invalidate cache when portfolio changes"""
    # Clear relevant cache entries
    pass
```

## Implementation Notes

### Data Consistency

**Transaction Synchronization**:
- Performance calculations update with each transaction
- Historical snapshots maintain data integrity
- Real-time calculations use latest market prices

### Error Handling

**Performance Calculation Safeguards**:
```python
def safe_calculate_return(current_value, invested_value):
    """Safely calculate returns with error handling"""
    try:
        if invested_value == 0:
            return 0.0
        return (current_value - invested_value) / invested_value
    except (ZeroDivisionError, TypeError):
        return 0.0
```

### Precision Management

**Decimal Precision for Performance**:
```python
from decimal import Decimal, ROUND_HALF_UP

def precise_performance_calculation(current_value, invested_value):
    """Calculate performance with decimal precision"""
    current = Decimal(str(current_value))
    invested = Decimal(str(invested_value))
    
    if invested == 0:
        return Decimal('0.00')
    
    return ((current - invested) / invested).quantize(
        Decimal('0.0001'), rounding=ROUND_HALF_UP
    )
```
