# Portfolio Metrics Documentation

This document details all portfolio-related metrics and calculations implemented in OptiTrade.

## Core Portfolio Metrics

### 1. Net Worth Calculation

**Formula**: `Net Worth = Cash Balance + Portfolio Value`

**Implementation**: 
```python
user_balance.net_worth = (new_cash_balance + new_portfolio_value).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Description**: Total value of user's account including cash and investments.

**Location**: `scripts/portfolio_management.py` (lines 410-412, 478-480, 552-554)

### 2. Portfolio Value

**Formula**: `Portfolio Value = Σ(Quantity × Current Price)` for all holdings

**Implementation**:
```python
portfolio_item.current_value = new_quantity * current_price
new_portfolio_value = (u_portfolio + quantity * current_price).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Description**: Current market value of all stock positions in the portfolio.

**Updates**: Real-time with market price changes

### 3. Average Price Calculation

**Formula**: `Average Price = (Previous Total Cost + New Purchase Cost) / Total Quantity`

**Implementation**:
```python
portfolio_item.average_price = (
    (portfolio_quantity * portfolio_avg_price + total_cost) / new_quantity
).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**Description**: Weighted average cost basis for each stock position.

**Use Case**: Tracks the average purchase price across multiple buy transactions.

### 4. Total Invested

**Formula**: `Total Invested = Quantity × Average Price`

**Implementation**:
```python
portfolio_item.total_invested = (
    portfolio_total_invested + total_cost
).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

**Description**: Total amount of money invested in a particular stock position.

### 5. Current Value per Position

**Formula**: `Current Value = Quantity × Current Market Price`

**Implementation**:
```python
portfolio_item.current_value = new_quantity * current_price
```

**Description**: Real-time market value of individual stock positions.

## Transaction Impact Calculations

### Buy Transaction Impact

**Cash Balance Update**:
```python
new_cash_balance = (u_balance - total_cost).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Portfolio Value Update**:
```python
new_portfolio_value = (u_portfolio + quantity * current_price).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

### Sell Transaction Impact

**Cash Balance Update**:
```python
new_cash_balance = (u_balance + total_sale).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Portfolio Value Update**:
```python
new_portfolio_value = (u_portfolio - quantity * current_price).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

**Position Adjustment**:
```python
if new_quantity > 0:
    portfolio_item.total_invested = (
        new_quantity * portfolio_item.average_price
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
else:
    # Position completely sold - remove from portfolio
    await db.delete(portfolio_item)
```

## Portfolio History Tracking

### Historical Snapshots

**Implementation**: `scripts/portfolio_history.py`

**Process**:
1. Daily snapshot of all portfolio positions
2. Stores historical values for trend analysis
3. Prevents duplicate entries for the same date

**Data Captured**:
- User ID
- Stock symbol
- Quantity held
- Average price
- Current value
- Total invested
- Snapshot date

### Portfolio History Retrieval

**API Endpoint**: `/portfolio/history/{user_id}`

**Features**:
- Filter by specific stock symbol
- Ordered by date (most recent first)
- Supports pagination

## Data Models

### UserBalanceModel
```python
class UserBalanceModel(Base):
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    cash_balance = Column(Float, nullable=False)
    portfolio_value = Column(Float, nullable=False, default=0.00)
    net_worth = Column(Float, nullable=False)
```

### Portfolio Model
```python
class Portfolio(Base):
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    symbol = Column(String(7), primary_key=True)
    quantity = Column(Integer, nullable=False)
    average_price = Column(DECIMAL(10, 2), nullable=False)
    current_value = Column(Float, nullable=False)
    total_invested = Column(DECIMAL(10, 2), nullable=False)
```

## Precision and Rounding

All monetary calculations use:
- **Decimal type** for precision
- **ROUND_HALF_UP** rounding method
- **2 decimal places** for currency values

This ensures accurate financial calculations without floating-point precision errors.

## Real-time Updates

Portfolio metrics are updated in real-time:
- **Price Changes**: Portfolio values recalculate with market price updates
- **Transactions**: Immediate impact on all relevant metrics
- **Balance Updates**: Synchronized across cash and portfolio values

## Error Handling

The system includes comprehensive error handling:
- **Insufficient Funds**: Prevents transactions exceeding available cash
- **Invalid Quantities**: Validates sell quantities against holdings
- **Data Integrity**: Ensures consistent state across all metrics
