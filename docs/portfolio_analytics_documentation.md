# Portfolio Analytics API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Analytics Metrics](#analytics-metrics)
4. [API Endpoints](#api-endpoints)
5. [Request/Response Examples](#request-response-examples)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Overview

The Portfolio Analytics API provides comprehensive risk and performance analysis for stock portfolios. This module calculates key financial metrics that help investors understand their portfolio's risk profile, performance characteristics, and diversification levels.

### Key Features
- **Value at Risk (VaR)**: Estimate potential portfolio losses
- **Maximum Drawdown (MDD)**: Measure worst-case decline scenarios
- **Sharpe Ratio**: Evaluate risk-adjusted returns
- **Beta**: Assess market sensitivity
- **Portfolio Concentration**: Analyze diversification levels

### Technology Stack
- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Analytics**: NumPy, SciPy, Pandas
- **Data Source**: Yahoo Finance (yfinance)

## Getting Started

### Prerequisites
- Python 3.13+
- PostgreSQL database
- Required Python packages (see requirements.txt)

### Installation
```bash
pip install -r requirements.txt
```

### Base URL
```
http://localhost:8000/analytics
```

## Analytics Metrics

### 1. Value at Risk (VaR)

**Definition**: VaR estimates the potential loss in portfolio value over a specific time period at a given confidence level.

**Calculation Method**: Historical simulation using portfolio returns
- Uses historical portfolio returns to estimate future risk
- Calculates percentile-based VaR at specified confidence level
- Scales returns for different time horizons

**Interpretation**:
- 95% VaR of -2.5% means there's a 5% chance of losing more than 2.5% in one day
- Higher absolute VaR values indicate higher risk
- Commonly used confidence levels: 95%, 99%

**Use Cases**:
- Risk management and position sizing
- Regulatory capital requirements
- Portfolio optimization

### 2. Maximum Drawdown (MDD)

**Definition**: MDD measures the largest peak-to-trough decline in portfolio value over a specified period.

**Calculation Method**: 
- Identifies the highest portfolio value (peak)
- Finds the lowest subsequent value (trough)
- Calculates percentage decline from peak to trough

**Interpretation**:
- MDD of -15% means the portfolio declined 15% from its peak
- Lower absolute values indicate better downside protection
- Helps assess worst-case historical scenarios

**Use Cases**:
- Risk assessment and stress testing
- Performance evaluation
- Investment strategy comparison

### 3. Sharpe Ratio

**Definition**: The Sharpe ratio measures risk-adjusted returns by comparing excess returns to volatility.

**Calculation Method**:
```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```

**Interpretation**:
- Higher values indicate better risk-adjusted performance
- Sharpe > 1.0 is generally considered good
- Sharpe > 2.0 is considered excellent
- Negative values indicate returns below risk-free rate

**Use Cases**:
- Performance comparison across portfolios
- Investment strategy evaluation
- Risk-adjusted return optimization

### 4. Beta

**Definition**: Beta measures portfolio sensitivity to market movements relative to a benchmark index.

**Calculation Method**: Linear regression of portfolio returns against market returns
```
Beta = Covariance(Portfolio, Market) / Variance(Market)
```

**Interpretation**:
- Beta = 1.0: Portfolio moves with the market
- Beta > 1.0: Portfolio is more volatile than market
- Beta < 1.0: Portfolio is less volatile than market
- Beta < 0: Portfolio moves opposite to market

**Use Cases**:
- Market risk assessment
- Portfolio construction and hedging
- Performance attribution analysis

### 5. Portfolio Concentration (Herfindahl-Hirschman Index)

**Definition**: HHI measures portfolio diversification by calculating the sum of squared position weights.

**Calculation Method**:
```
HHI = Σ(wi²) where wi is the weight of position i
```

**Interpretation**:
- HHI < 0.15: Low concentration (well diversified)
- HHI 0.15-0.25: Moderate concentration
- HHI > 0.25: High concentration (concentrated portfolio)
- Lower values indicate better diversification

**Use Cases**:
- Diversification analysis
- Risk management
- Regulatory compliance

## API Endpoints

### Base Endpoint Information
- **Base URL**: `/analytics`
- **Authentication**: Required (user_id in request body for POST, path parameter for GET)
- **Content-Type**: `application/json` (for POST requests)
- **Response Format**: JSON

### HTTP Methods
All analytics endpoints support both **POST** and **GET** methods:

#### POST Method
- **Use Case**: When you need to send complex parameters or prefer request body format
- **Parameters**: Sent in JSON request body
- **Example**: `POST /analytics/var` with JSON body

#### GET Method
- **Use Case**: Simple requests, URL bookmarking, caching, or when request body is not preferred
- **Parameters**: Sent as URL path parameters and query parameters
- **Example**: `GET /analytics/var/1?confidence_level=0.95`

**Recommendation**: Use GET for simple requests and POST for complex parameter sets or when integrating with systems that prefer request bodies.

### 1. Calculate Value at Risk

#### POST Method
```http
POST /analytics/var
```

**Request Body**:
```json
{
  "user_id": 1,
  "confidence_level": 0.95,
  "time_horizon_days": 1,
  "historical_days": 252
}
```

#### GET Method
```http
GET /analytics/var/{user_id}?confidence_level=0.95&time_horizon_days=1&historical_days=252
```

**Example**:
```http
GET /analytics/var/1?confidence_level=0.95&time_horizon_days=1&historical_days=252
```

**Response** (both methods):
```json
{
  "var_percentage": -0.025,
  "var_dollar": 2500.00,
  "confidence_level": 0.95,
  "time_horizon_days": 1,
  "current_portfolio_value": 100000.00,
  "mean_daily_return": 0.001,
  "daily_volatility": 0.015,
  "observations_used": 252
}
```

### 2. Calculate Maximum Drawdown

#### POST Method
```http
POST /analytics/max-drawdown
```

**Request Body**:
```json
{
  "user_id": 1,
  "historical_days": 252
}
```

#### GET Method
```http
GET /analytics/max-drawdown/{user_id}?historical_days=252
```

**Example**:
```http
GET /analytics/max-drawdown/1?historical_days=252
```

**Response** (both methods):
```json
{
  "max_drawdown_percentage": -0.15,
  "peak_date": "2024-01-15T00:00:00",
  "trough_date": "2024-03-10T00:00:00",
  "recovery_date": "2024-05-20T00:00:00",
  "peak_value": 120000.00,
  "trough_value": 102000.00,
  "current_value": 115000.00,
  "drawdown_duration_days": 55,
  "observations_used": 252
}
```

### 3. Calculate Sharpe Ratio

#### POST Method
```http
POST /analytics/sharpe-ratio
```

**Request Body**:
```json
{
  "user_id": 1,
  "risk_free_rate": 0.02,
  "historical_days": 252
}
```

#### GET Method
```http
GET /analytics/sharpe-ratio/{user_id}?risk_free_rate=0.02&historical_days=252
```

**Example**:
```http
GET /analytics/sharpe-ratio/1?risk_free_rate=0.02&historical_days=252
```

**Response** (both methods):
```json
{
  "sharpe_ratio": 1.25,
  "annualized_return": 0.12,
  "annualized_volatility": 0.08,
  "risk_free_rate": 0.02,
  "excess_return": 0.10,
  "observations_used": 252
}
```

### 4. Calculate Beta

#### POST Method
```http
POST /analytics/beta
```

**Request Body**:
```json
{
  "user_id": 1,
  "market_symbol": "SPY",
  "historical_days": 252
}
```

#### GET Method
```http
GET /analytics/beta/{user_id}?market_symbol=SPY&historical_days=252
```

**Example**:
```http
GET /analytics/beta/1?market_symbol=SPY&historical_days=252
```

**Response** (both methods):
```json
{
  "beta": 1.15,
  "alpha_annualized": 0.02,
  "r_squared": 0.75,
  "correlation": 0.87,
  "portfolio_volatility": 0.18,
  "market_volatility": 0.16,
  "market_symbol": "SPY",
  "p_value": 0.001,
  "observations_used": 252
}
```

### 5. Calculate Portfolio Concentration

#### POST Method
```http
POST /analytics/concentration
```

**Request Body**:
```json
{
  "user_id": 1
}
```

#### GET Method
```http
GET /analytics/concentration/{user_id}
```

**Example**:
```http
GET /analytics/concentration/1
```

**Response** (both methods):
```json
{
  "herfindahl_hirschman_index": 0.18,
  "effective_number_of_holdings": 5.56,
  "actual_number_of_holdings": 10,
  "concentration_level": "Moderate",
  "total_portfolio_value": 100000.00,
  "top_5_holdings_weight": 0.65,
  "top_holdings": [
    {
      "symbol": "AAPL",
      "value": 25000.00,
      "weight": 0.25,
      "quantity": 100
    }
  ],
  "all_holdings": []
}
```

### 6. Comprehensive Analytics

#### POST Method
```http
POST /analytics/comprehensive
```

**Request Body**:
```json
{
  "user_id": 1,
  "confidence_level": 0.95,
  "risk_free_rate": 0.02,
  "market_symbol": "SPY",
  "historical_days": 252
}
```

#### GET Method
```http
GET /analytics/comprehensive/{user_id}?confidence_level=0.95&risk_free_rate=0.02&market_symbol=SPY&historical_days=252
```

**Example**:
```http
GET /analytics/comprehensive/1?confidence_level=0.95&risk_free_rate=0.02&market_symbol=SPY&historical_days=252
```

**Response** (both methods): Combined response containing all analytics metrics.

### 7. Health Check
```http
GET /analytics/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "portfolio_analytics",
  "version": "1.0.0"
}
```

## Error Handling

### Common Error Codes

#### 400 Bad Request
Returned when request parameters are invalid or insufficient data is available.

**Example Response**:
```json
{
  "detail": "Insufficient historical data for VaR calculation"
}
```

**Common Causes**:
- User has no portfolio data
- Insufficient historical data (< 30 observations)
- Invalid confidence levels or parameters
- Portfolio has no positive value

#### 404 Not Found
Returned when user or portfolio data is not found.

**Example Response**:
```json
{
  "detail": "Portfolio data not found for user"
}
```

#### 500 Internal Server Error
Returned when unexpected server errors occur.

**Example Response**:
```json
{
  "detail": "Internal server error calculating VaR"
}
```

### Error Handling Best Practices

1. **Check Response Status**: Always check HTTP status codes
2. **Handle Partial Failures**: Comprehensive endpoint may return partial results
3. **Retry Logic**: Implement exponential backoff for temporary failures
4. **Validate Input**: Ensure parameters are within valid ranges

## Best Practices

### Data Requirements

#### Minimum Data Requirements
- **VaR**: At least 30 days of portfolio history
- **Maximum Drawdown**: At least 2 days of portfolio history
- **Sharpe Ratio**: At least 30 days of portfolio history
- **Beta**: At least 30 days of overlapping portfolio and market data
- **Concentration**: Current portfolio holdings

#### Recommended Data Periods
- **Short-term analysis**: 63 days (3 months)
- **Standard analysis**: 252 days (1 year)
- **Long-term analysis**: 504 days (2 years)

### Parameter Guidelines

#### VaR Parameters
- **Confidence Level**:
  - 95% for standard risk management
  - 99% for conservative risk management
  - 90% for aggressive strategies
- **Time Horizon**:
  - 1 day for daily risk monitoring
  - 10 days for regulatory reporting
  - 21 days for monthly risk assessment

#### Beta Parameters
- **Market Symbol**:
  - SPY for US large-cap exposure
  - QQQ for technology-heavy portfolios
  - IWM for small-cap exposure
  - VTI for total market exposure

#### Risk-Free Rate
- Use current 3-month Treasury rate
- Update quarterly for accuracy
- Consider using FRED API for real-time rates

### Performance Optimization

#### Caching Strategies
- Cache market data for multiple users
- Store calculated metrics with timestamps
- Implement Redis for high-frequency requests

#### Batch Processing
- Use comprehensive endpoint for multiple metrics
- Process multiple users in parallel
- Schedule regular metric updates

### Integration Examples

#### Python Client Example
```python
import requests
import json

class PortfolioAnalytics:
    def __init__(self, base_url):
        self.base_url = base_url

    def calculate_var_post(self, user_id, confidence_level=0.95):
        """Calculate VaR using POST method"""
        url = f"{self.base_url}/analytics/var"
        payload = {
            "user_id": user_id,
            "confidence_level": confidence_level,
            "time_horizon_days": 1,
            "historical_days": 252
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def calculate_var_get(self, user_id, confidence_level=0.95):
        """Calculate VaR using GET method"""
        url = f"{self.base_url}/analytics/var/{user_id}"
        params = {
            "confidence_level": confidence_level,
            "time_horizon_days": 1,
            "historical_days": 252
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_comprehensive_analytics_post(self, user_id):
        """Get comprehensive analytics using POST method"""
        url = f"{self.base_url}/analytics/comprehensive"
        payload = {"user_id": user_id}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_comprehensive_analytics_get(self, user_id):
        """Get comprehensive analytics using GET method"""
        url = f"{self.base_url}/analytics/comprehensive/{user_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# Usage
analytics = PortfolioAnalytics("http://localhost:8000")

# Using POST method
var_result_post = analytics.calculate_var_post(user_id=1)
print(f"Portfolio VaR (POST): {var_result_post['var_percentage']:.2%}")

# Using GET method
var_result_get = analytics.calculate_var_get(user_id=1)
print(f"Portfolio VaR (GET): {var_result_get['var_percentage']:.2%}")
```

#### JavaScript Client Example
```javascript
class PortfolioAnalytics {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async calculateVaRPost(userId, confidenceLevel = 0.95) {
        const response = await fetch(`${this.baseUrl}/analytics/var`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                confidence_level: confidenceLevel,
                time_horizon_days: 1,
                historical_days: 252
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async calculateVaRGet(userId, confidenceLevel = 0.95) {
        const params = new URLSearchParams({
            confidence_level: confidenceLevel,
            time_horizon_days: 1,
            historical_days: 252
        });

        const response = await fetch(`${this.baseUrl}/analytics/var/${userId}?${params}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async getComprehensiveAnalyticsPost(userId) {
        const response = await fetch(`${this.baseUrl}/analytics/comprehensive`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async getComprehensiveAnalyticsGet(userId) {
        const response = await fetch(`${this.baseUrl}/analytics/comprehensive/${userId}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
}

// Usage
const analytics = new PortfolioAnalytics('http://localhost:8000');

// Using POST method
analytics.calculateVaRPost(1)
    .then(result => console.log(`Portfolio VaR (POST): ${(result.var_percentage * 100).toFixed(2)}%`))
    .catch(error => console.error('Error:', error));

// Using GET method
analytics.calculateVaRGet(1)
    .then(result => console.log(`Portfolio VaR (GET): ${(result.var_percentage * 100).toFixed(2)}%`))
    .catch(error => console.error('Error:', error));
```

## Appendix

### Mathematical Formulas

#### Value at Risk (Historical Simulation)
```
VaR(α) = -Percentile(Returns, (1-α) × 100)
```
Where α is the confidence level (e.g., 0.95 for 95% confidence)

#### Maximum Drawdown
```
DD(t) = (V(t) - Peak(t)) / Peak(t)
MDD = min(DD(t)) for all t
```
Where V(t) is portfolio value at time t, Peak(t) is running maximum

#### Sharpe Ratio
```
Sharpe = (Rp - Rf) / σp
```
Where Rp is portfolio return, Rf is risk-free rate, σp is portfolio volatility

#### Beta
```
β = Cov(Rp, Rm) / Var(Rm)
```
Where Rp is portfolio returns, Rm is market returns

#### Herfindahl-Hirschman Index
```
HHI = Σ(wi²) for i = 1 to n
```
Where wi is the weight of asset i in the portfolio

### Glossary

- **Alpha**: Risk-adjusted excess return relative to a benchmark
- **Correlation**: Statistical measure of how two assets move together
- **Drawdown**: Peak-to-trough decline in portfolio value
- **R-squared**: Proportion of variance explained by the market
- **Volatility**: Standard deviation of returns, measuring price variability

### Support and Contact

For technical support or questions about the Portfolio Analytics API:
- Email: support@optitrade.com
- Documentation: https://docs.optitrade.com
- API Status: https://status.optitrade.com

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-17
**API Version**: 1.0.0
