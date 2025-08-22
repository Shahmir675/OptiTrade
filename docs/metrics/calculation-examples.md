# Calculation Examples Documentation

This document provides practical examples of how metrics are calculated in OptiTrade, with step-by-step breakdowns and real-world scenarios.

## Portfolio Metrics Examples

### Example 1: Basic Portfolio Calculation

**Scenario**: User buys 100 shares of AAPL at $150.00, then buys 50 more shares at $160.00

**Step 1: First Purchase**
```python
# Initial purchase
quantity_1 = 100
price_1 = Decimal('150.00')
total_cost_1 = quantity_1 * price_1  # $15,000.00

# Portfolio state after first purchase
portfolio_item = Portfolio(
    user_id=1,
    symbol='AAPL',
    quantity=100,
    average_price=Decimal('150.00'),
    current_value=100 * 150.00,  # $15,000.00
    total_invested=Decimal('15000.00')
)
```

**Step 2: Second Purchase (Average Price Calculation)**
```python
# Second purchase
quantity_2 = 50
price_2 = Decimal('160.00')
total_cost_2 = quantity_2 * price_2  # $8,000.00

# Calculate new average price
new_quantity = 100 + 50  # 150 shares
new_average_price = (
    (100 * Decimal('150.00') + 50 * Decimal('160.00')) / 150
).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
# = (15,000 + 8,000) / 150 = $153.33

# Updated portfolio state
portfolio_item.quantity = 150
portfolio_item.average_price = Decimal('153.33')
portfolio_item.total_invested = Decimal('23000.00')
portfolio_item.current_value = 150 * current_market_price
```

### Example 2: Portfolio Value and Net Worth Calculation

**Scenario**: User has multiple positions and cash balance

**Portfolio Holdings**:
- AAPL: 150 shares at $155.00 current price (avg cost $153.33)
- GOOGL: 25 shares at $2,800.00 current price (avg cost $2,750.00)
- Cash Balance: $5,000.00

**Calculations**:
```python
# Individual position values
aapl_current_value = 150 * Decimal('155.00')  # $23,250.00
googl_current_value = 25 * Decimal('2800.00')  # $70,000.00

# Total portfolio value
portfolio_value = aapl_current_value + googl_current_value  # $93,250.00

# Net worth calculation
cash_balance = Decimal('5000.00')
net_worth = cash_balance + portfolio_value  # $98,250.00

# User balance update
user_balance.cash_balance = Decimal('5000.00')
user_balance.portfolio_value = Decimal('93250.00')
user_balance.net_worth = Decimal('98250.00')
```

## Performance Metrics Examples

### Example 3: Return Calculations

**Using the AAPL position from Example 1**:

**Unrealized Return Calculation**:
```python
# Position details
total_invested = Decimal('23000.00')  # Total amount invested
current_value = 150 * Decimal('155.00')  # $23,250.00

# Unrealized gain/loss
unrealized_pnl = current_value - total_invested  # $250.00
unrealized_return = (unrealized_pnl / total_invested) * 100  # 1.09%

print(f"Unrealized P&L: ${unrealized_pnl}")
print(f"Unrealized Return: {unrealized_return:.2f}%")
```

**Portfolio-Level Return**:
```python
# Total portfolio calculation
total_invested_portfolio = Decimal('23000.00') + Decimal('68750.00')  # $91,750.00
total_current_value = Decimal('23250.00') + Decimal('70000.00')  # $93,250.00

portfolio_return = (
    (total_current_value - total_invested_portfolio) / total_invested_portfolio
) * 100  # 1.63%

print(f"Portfolio Return: {portfolio_return:.2f}%")
```

### Example 4: Sell Transaction and Realized Gains

**Scenario**: User sells 50 shares of AAPL at $158.00

**Calculation**:
```python
# Sell transaction details
sell_quantity = 50
sell_price = Decimal('158.00')
total_sale = sell_quantity * sell_price  # $7,900.00

# Calculate realized gain (FIFO method)
cost_basis = sell_quantity * portfolio_item.average_price  # 50 * $153.33 = $7,666.50
realized_gain = total_sale - cost_basis  # $7,900.00 - $7,666.50 = $233.50

# Update portfolio position
new_quantity = 150 - 50  # 100 shares remaining
portfolio_item.quantity = new_quantity
portfolio_item.total_invested = new_quantity * portfolio_item.average_price  # $15,333.00
portfolio_item.current_value = new_quantity * current_market_price

# Update cash balance
new_cash_balance = current_cash_balance + total_sale
```

## Risk Metrics Examples

### Example 5: Volatility Calculation

**Scenario**: Calculate 30-day annualized volatility for AAPL

**Historical Price Data** (last 30 days of closing prices):
```python
import numpy as np
import pandas as pd

# Sample closing prices (simplified)
closing_prices = pd.Series([
    150.00, 151.20, 149.80, 152.10, 153.50, 152.90, 154.20, 155.10,
    153.80, 156.20, 157.40, 156.80, 158.90, 157.20, 159.10, 160.50,
    # ... 30 days of data
])

# Step 1: Calculate log returns
log_returns = np.log(closing_prices / closing_prices.shift(1)).dropna()

# Step 2: Calculate rolling standard deviation
rolling_window = 30
vol_series = log_returns.rolling(window=rolling_window).std().dropna()

# Step 3: Annualize volatility
annualized_volatility = vol_series.iloc[-1] * np.sqrt(252)  # 252 trading days

print(f"30-day Annualized Volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
```

### Example 6: Portfolio Concentration Risk

**Scenario**: Assess concentration risk in a portfolio

**Portfolio Composition**:
```python
portfolio_positions = [
    {'symbol': 'AAPL', 'value': 23250.00},
    {'symbol': 'GOOGL', 'value': 70000.00},
    {'symbol': 'MSFT', 'value': 15000.00},
    {'symbol': 'TSLA', 'value': 8000.00}
]

total_portfolio_value = sum(pos['value'] for pos in portfolio_positions)  # $116,250.00

# Calculate position weights
for position in portfolio_positions:
    weight = position['value'] / total_portfolio_value
    position['weight'] = weight
    print(f"{position['symbol']}: ${position['value']:,.2f} ({weight:.2%})")

# Identify concentration risk
max_position = max(portfolio_positions, key=lambda x: x['weight'])
concentration_ratio = max_position['weight']

if concentration_ratio > 0.30:  # 30% threshold
    print(f"⚠️  High concentration risk: {max_position['symbol']} represents {concentration_ratio:.2%}")
```

**Output**:
```
AAPL: $23,250.00 (20.00%)
GOOGL: $70,000.00 (60.22%)
MSFT: $15,000.00 (12.91%)
TSLA: $8,000.00 (6.88%)
⚠️  High concentration risk: GOOGL represents 60.22%
```

## Technical Indicators Examples

### Example 7: Moving Average Calculations

**Scenario**: Calculate SMA20, SMA50, and SMA200 for trend analysis

```python
# Sample price data (last 200 days)
price_data = pd.Series([...])  # 200 days of closing prices

# Calculate moving averages
sma_20 = price_data.rolling(window=20).mean().iloc[-1]
sma_50 = price_data.rolling(window=50).mean().iloc[-1]
sma_200 = price_data.rolling(window=200).mean().iloc[-1]

current_price = price_data.iloc[-1]

# Trend analysis
def analyze_trend(current, sma20, sma50, sma200):
    if current > sma20 > sma50 > sma200:
        return "Strong Uptrend"
    elif current > sma20 > sma50:
        return "Uptrend"
    elif current < sma20 < sma50 < sma200:
        return "Strong Downtrend"
    elif current < sma20 < sma50:
        return "Downtrend"
    else:
        return "Sideways/Mixed"

trend = analyze_trend(current_price, sma_20, sma_50, sma_200)
print(f"Current Price: ${current_price:.2f}")
print(f"SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f}, SMA200: ${sma_200:.2f}")
print(f"Trend: {trend}")
```

### Example 8: RSI Calculation

**Scenario**: Calculate 14-period RSI for momentum analysis

```python
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI
price_series = pd.Series([...])  # Price data
rsi_values = calculate_rsi(price_series)
current_rsi = rsi_values.iloc[-1]

# RSI interpretation
def interpret_rsi(rsi):
    if rsi > 70:
        return "Overbought"
    elif rsi < 30:
        return "Oversold"
    else:
        return "Neutral"

interpretation = interpret_rsi(current_rsi)
print(f"Current RSI: {current_rsi:.2f} - {interpretation}")
```

## Market Data Examples

### Example 9: NASDAQ Index Performance

**Scenario**: Calculate daily NASDAQ performance

```python
# Market data
current_nasdaq = 15234.56
previous_close = 15180.23

# Daily performance calculation
daily_change = current_nasdaq - previous_close  # 54.33
percent_change = (daily_change / previous_close) * 100  # 0.36%

# Format for display
performance_data = {
    "current": round(current_nasdaq, 2),
    "previous_close": round(previous_close, 2),
    "daily_change": round(daily_change, 2),
    "percent_change": round(percent_change, 2)
}

print(f"NASDAQ: {performance_data['current']} ({performance_data['daily_change']:+.2f}, {performance_data['percent_change']:+.2f}%)")
```

### Example 10: Volume Analysis

**Scenario**: Analyze unusual volume activity

```python
# Volume data
current_volume = 2500000
average_volume_30d = 1800000

# Relative volume calculation
relative_volume = current_volume / average_volume_30d  # 1.39

# Volume analysis
def analyze_volume(rel_vol):
    if rel_vol > 2.0:
        return "Extremely High Volume"
    elif rel_vol > 1.5:
        return "High Volume"
    elif rel_vol > 1.2:
        return "Above Average Volume"
    elif rel_vol < 0.5:
        return "Low Volume"
    else:
        return "Normal Volume"

volume_analysis = analyze_volume(relative_volume)
print(f"Volume: {current_volume:,} (Avg: {average_volume_30d:,})")
print(f"Relative Volume: {relative_volume:.2f}x - {volume_analysis}")
```

## Financial Ratios Examples

### Example 11: P/E Ratio Analysis

**Scenario**: Evaluate stock valuation using P/E ratios

```python
# Company financial data
current_price = 155.00
earnings_per_share = 6.15
forward_eps_estimate = 6.80

# P/E calculations
trailing_pe = current_price / earnings_per_share  # 25.20
forward_pe = current_price / forward_eps_estimate  # 22.79

# Valuation assessment
def assess_valuation(pe_ratio, industry_avg_pe=20.0):
    premium_discount = ((pe_ratio - industry_avg_pe) / industry_avg_pe) * 100
    
    if pe_ratio > industry_avg_pe * 1.2:
        return f"Expensive ({premium_discount:+.1f}% vs industry)"
    elif pe_ratio < industry_avg_pe * 0.8:
        return f"Cheap ({premium_discount:+.1f}% vs industry)"
    else:
        return f"Fair Value ({premium_discount:+.1f}% vs industry)"

valuation = assess_valuation(trailing_pe)
print(f"Trailing P/E: {trailing_pe:.2f}")
print(f"Forward P/E: {forward_pe:.2f}")
print(f"Valuation: {valuation}")
```

## Transaction Cost Examples

### Example 12: Transaction Impact Analysis

**Scenario**: Calculate the impact of a large buy order

```python
# Order details
symbol = "AAPL"
order_quantity = 1000
current_price = Decimal('155.00')
user_cash_balance = Decimal('200000.00')

# Transaction cost calculation
total_cost = order_quantity * current_price  # $155,000.00

# Validate sufficient funds
if user_cash_balance < total_cost:
    raise Exception(f"Insufficient funds. Required: ${total_cost}, Available: ${user_cash_balance}")

# Calculate portfolio impact
current_portfolio_value = Decimal('300000.00')
new_portfolio_value = current_portfolio_value + total_cost
position_weight = total_cost / new_portfolio_value  # 33.33%

# Risk assessment
if position_weight > 0.25:  # 25% concentration limit
    print(f"⚠️  Warning: Position will represent {position_weight:.2%} of portfolio")

# Update balances
new_cash_balance = user_cash_balance - total_cost  # $45,000.00
new_net_worth = new_cash_balance + new_portfolio_value  # $500,000.00

print(f"Transaction Cost: ${total_cost:,.2f}")
print(f"New Cash Balance: ${new_cash_balance:,.2f}")
print(f"New Portfolio Value: ${new_portfolio_value:,.2f}")
print(f"Position Weight: {position_weight:.2%}")
```

## Error Handling Examples

### Example 13: Safe Calculation with Error Handling

**Scenario**: Robust return calculation with error handling

```python
def safe_calculate_return(current_value, invested_value):
    """Safely calculate returns with comprehensive error handling"""
    try:
        # Convert to Decimal for precision
        current = Decimal(str(current_value)) if current_value is not None else Decimal('0')
        invested = Decimal(str(invested_value)) if invested_value is not None else Decimal('0')
        
        # Handle zero investment
        if invested == 0:
            return Decimal('0.00')
        
        # Calculate return
        return_value = ((current - invested) / invested).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )
        
        return float(return_value)
        
    except (ValueError, TypeError, InvalidOperation) as e:
        print(f"Error calculating return: {e}")
        return 0.00
    except ZeroDivisionError:
        print("Division by zero in return calculation")
        return 0.00

# Usage examples
print(safe_calculate_return(23250.00, 23000.00))  # 0.0109 (1.09%)
print(safe_calculate_return(15000.00, 0))         # 0.00 (handles zero investment)
print(safe_calculate_return(None, 15000.00))      # 0.00 (handles None values)
```

These examples demonstrate the practical implementation of OptiTrade's metrics calculations, showing how the theoretical formulas translate into working code with proper error handling and precision management.
