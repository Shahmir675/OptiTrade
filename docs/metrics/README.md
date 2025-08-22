# OptiTrade Metrics and Calculations Documentation

This section provides comprehensive documentation of all financial metrics, calculations, and performance indicators implemented in the OptiTrade platform.

## Table of Contents

1. [Portfolio Metrics](./portfolio-metrics.md)
2. [Financial Ratios](./financial-ratios.md)
3. [Technical Indicators](./technical-indicators.md)
4. [Risk Metrics](./risk-metrics.md)
5. [Performance Metrics](./performance-metrics.md)
6. [Market Data Metrics](./market-data-metrics.md)
7. [Calculation Examples](./calculation-examples.md)

## Overview

OptiTrade implements a comprehensive suite of financial metrics and calculations to provide users with detailed insights into their investments and market conditions. These metrics are categorized into several key areas:

### Portfolio Metrics
- **Net Worth Calculation**: Total portfolio value plus cash balance
- **Portfolio Value**: Current market value of all holdings
- **Average Price Calculation**: Weighted average cost basis
- **Total Investment**: Total amount invested across all positions
- **Current Value**: Real-time valuation of portfolio positions

### Financial Analysis
- **Profit/Loss Calculations**: Realized and unrealized gains/losses
- **Return Calculations**: Various return metrics and time periods
- **Cost Basis Tracking**: Average cost per share calculations
- **Transaction Analysis**: Buy/sell transaction metrics

### Risk Assessment
- **Volatility Calculations**: Historical volatility analysis
- **Portfolio Concentration**: Position sizing and diversification metrics
- **Market Risk Indicators**: Beta, correlation, and risk-adjusted returns

### Technical Indicators
- **Moving Averages**: SMA20, SMA50, SMA200
- **Momentum Indicators**: RSI, performance metrics
- **Volatility Measures**: ATR, volatility windows
- **Price Relationships**: Price to moving average ratios

## Data Sources

The platform integrates multiple data sources for comprehensive market analysis:

- **Yahoo Finance**: Primary source for stock prices and historical data
- **Finviz**: Financial ratios, technical indicators, and screening data
- **Real-time Feeds**: Live price updates and market data

## Calculation Precision

All financial calculations use decimal precision to ensure accuracy:
- Monetary values: 2 decimal places
- Percentages: 2-4 decimal places depending on context
- Ratios: 4 decimal places for precision

## Implementation Details

Metrics are implemented across several modules:
- `scripts/portfolio_management.py`: Core portfolio calculations
- `scripts/fetch_nasdaq.py`: Market data and volatility calculations
- `finvizfinance/`: Financial ratios and technical indicators
- `app/models/`: Data models for metric storage

## Real-time Updates

Many metrics are calculated in real-time:
- Portfolio values update with current market prices
- Risk metrics recalculate with new market data
- Performance metrics update with each transaction

For detailed information on specific metrics, please refer to the individual documentation files in this section.
