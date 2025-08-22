# Risk Metrics Documentation

This document details all risk assessment metrics and calculations implemented in OptiTrade for portfolio and investment risk analysis.

## Overview

OptiTrade implements comprehensive risk metrics to help users understand and manage investment risks. These metrics cover volatility, market risk, concentration risk, and portfolio-level risk assessments.

## Volatility Metrics

### Historical Volatility

**Calculation Method**:
```python
# Log returns calculation
prices = hist["Close"].dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()

# Rolling volatility
vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()

# Annualized volatility
annualized_vol = vol_series.iloc[-1] * np.sqrt(252)
```

**Parameters**:
- **Rolling Window**: 30 days (configurable)
- **Annualization Factor**: √252 trading days
- **Location**: `scripts/fetch_nasdaq.py` (lines 149-154)

**Interpretation**:
- Higher volatility indicates greater price uncertainty
- Annualized volatility allows comparison across different time periods
- Used for risk-adjusted return calculations

### Volatility Windows

**Weekly Volatility (Volatility W)**
- **Description**: Short-term volatility measure
- **Use**: Identifies recent changes in price stability
- **Source**: `finvizfinance/constants.py` (line 46)

**Monthly Volatility (Volatility M)**
- **Description**: Medium-term volatility measure
- **Use**: Trend analysis of price stability
- **Source**: `finvizfinance/constants.py` (line 47)

## Market Risk Indicators

### Beta Coefficient

**Beta**
- **Description**: Measures systematic risk relative to market
- **Formula**: Covariance(Stock, Market) / Variance(Market)
- **Interpretation**:
  - β = 1.0: Stock moves with market
  - β > 1.0: Stock is more volatile than market
  - β < 1.0: Stock is less volatile than market
  - β < 0: Stock moves opposite to market
- **Source**: `finvizfinance/constants.py` (line 44)

**Applications**:
- Portfolio beta calculation
- Risk-adjusted return analysis
- Hedging strategy development
- Asset allocation decisions

### Average True Range (ATR)

**ATR (Average True Range)**
- **Formula**: Average of True Range over specified period
- **True Range**: Max of:
  - High - Low
  - |High - Previous Close|
  - |Low - Previous Close|
- **Use**: Measures price volatility independent of direction
- **Source**: `finvizfinance/constants.py` (line 45)

**Applications**:
- Position sizing calculations
- Stop-loss level determination
- Volatility-based trading strategies

## Portfolio Risk Metrics

### Concentration Risk

**Position Sizing Analysis**:
```python
# Calculate position weight in portfolio
position_weight = (quantity * current_price) / total_portfolio_value

# Monitor concentration limits
if position_weight > MAX_POSITION_WEIGHT:
    # Risk warning or position adjustment
```

**Metrics Tracked**:
- Individual position weights
- Sector concentration
- Geographic concentration
- Market cap concentration

### Portfolio Volatility

**Portfolio-Level Calculations**:
```python
# Portfolio value volatility
portfolio_returns = portfolio_values.pct_change().dropna()
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
```

**Components**:
- Individual asset volatilities
- Correlation between assets
- Position weights
- Diversification benefits

## Liquidity Risk Metrics

### Volume-Based Risk

**Average Volume Analysis**:
- **Avg Volume**: Historical average trading volume
- **Rel Volume**: Current volume relative to average
- **Use**: Assesses liquidity and market impact risk

**Implementation**:
```python
# Liquidity risk assessment
if avg_volume < MIN_LIQUIDITY_THRESHOLD:
    risk_level = "HIGH"
elif rel_volume < 0.5:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"
```

### Bid-Ask Spread Analysis

While not directly implemented, the system considers:
- Market impact of large orders
- Slippage estimation
- Execution risk assessment

## Financial Risk Indicators

### Leverage Risk

**Debt-to-Equity Ratios**:
- **LTDebt/Eq**: Long-term debt to equity ratio
- **Debt/Eq**: Total debt to equity ratio
- **Use**: Assesses financial leverage risk

**Risk Categories**:
```python
def assess_leverage_risk(debt_to_equity):
    if debt_to_equity > 2.0:
        return "HIGH"
    elif debt_to_equity > 1.0:
        return "MEDIUM"
    else:
        return "LOW"
```

### Liquidity Risk (Company Level)

**Current Ratio Analysis**:
- **Curr R**: Current assets / Current liabilities
- **Quick R**: (Current assets - Inventory) / Current liabilities

**Risk Assessment**:
```python
def assess_liquidity_risk(current_ratio, quick_ratio):
    if current_ratio < 1.0 or quick_ratio < 0.5:
        return "HIGH"
    elif current_ratio < 1.5 or quick_ratio < 1.0:
        return "MEDIUM"
    else:
        return "LOW"
```

## Risk-Adjusted Performance

### Sharpe Ratio Calculation

**Formula**: (Return - Risk-free Rate) / Standard Deviation

**Implementation**:
```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std()
```

### Maximum Drawdown

**Calculation**:
```python
def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()
```

## Risk Monitoring and Alerts

### Real-time Risk Assessment

**Portfolio Risk Monitoring**:
```python
async def assess_portfolio_risk(user_id, db):
    portfolio = await get_user_portfolio(user_id, db)
    
    # Calculate portfolio metrics
    total_value = sum(position.current_value for position in portfolio)
    
    # Concentration risk
    max_position = max(position.current_value for position in portfolio)
    concentration_ratio = max_position / total_value
    
    # Volatility risk
    portfolio_volatility = calculate_portfolio_volatility(portfolio)
    
    return {
        "concentration_risk": concentration_ratio,
        "volatility_risk": portfolio_volatility,
        "overall_risk": assess_overall_risk(concentration_ratio, portfolio_volatility)
    }
```

### Risk Thresholds

**Configurable Risk Limits**:
```python
RISK_THRESHOLDS = {
    "MAX_POSITION_WEIGHT": 0.20,  # 20% maximum position size
    "MAX_SECTOR_WEIGHT": 0.30,    # 30% maximum sector allocation
    "MAX_PORTFOLIO_VOLATILITY": 0.25,  # 25% maximum portfolio volatility
    "MIN_LIQUIDITY_VOLUME": 100000,    # Minimum daily volume
}
```

## Value at Risk (VaR)

### Historical VaR

**Calculation Method**:
```python
def calculate_historical_var(returns, confidence_level=0.05):
    """Calculate Historical Value at Risk"""
    return np.percentile(returns, confidence_level * 100)
```

### Parametric VaR

**Normal Distribution Assumption**:
```python
def calculate_parametric_var(returns, confidence_level=0.05):
    """Calculate Parametric VaR assuming normal distribution"""
    mean_return = returns.mean()
    std_return = returns.std()
    z_score = norm.ppf(confidence_level)
    return mean_return + z_score * std_return
```

## Stress Testing

### Scenario Analysis

**Market Stress Scenarios**:
- Market crash (-20% market decline)
- Interest rate shock (+200 basis points)
- Sector-specific stress tests
- Liquidity crisis scenarios

**Implementation Framework**:
```python
def stress_test_portfolio(portfolio, scenario):
    """Apply stress scenario to portfolio"""
    stressed_values = {}
    
    for position in portfolio:
        # Apply scenario-specific shocks
        shock_factor = scenario.get_shock_factor(position.symbol)
        stressed_value = position.current_value * (1 + shock_factor)
        stressed_values[position.symbol] = stressed_value
    
    return calculate_portfolio_impact(stressed_values)
```

## Risk Reporting

### Risk Dashboard Metrics

**Key Risk Indicators (KRIs)**:
1. Portfolio volatility
2. Maximum position concentration
3. Sector concentration
4. Liquidity risk score
5. Leverage exposure
6. Value at Risk (1-day, 95% confidence)

### Risk Alerts

**Automated Risk Monitoring**:
- Position concentration exceeds limits
- Portfolio volatility above threshold
- Liquidity concerns for holdings
- Significant drawdown alerts

## Implementation Notes

### Data Requirements

**Minimum Data for Risk Calculations**:
- 40+ days of price history for volatility
- Volume data for liquidity assessment
- Financial statement data for leverage analysis

### Calculation Frequency

**Update Schedules**:
- **Real-time**: Portfolio value and concentration
- **Daily**: Volatility and VaR calculations
- **Weekly**: Stress test scenarios
- **Monthly**: Comprehensive risk review

### Error Handling

**Risk Calculation Safeguards**:
```python
try:
    volatility = calculate_volatility(price_data)
    if volatility is None or volatility < 0:
        volatility = DEFAULT_VOLATILITY
except Exception as e:
    logger.warning(f"Volatility calculation failed: {e}")
    volatility = DEFAULT_VOLATILITY
```
