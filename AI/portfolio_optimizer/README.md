# Portfolio Optimizer with LSTM + PPO

A comprehensive portfolio optimization system that combines Long Short-Term Memory (LSTM) neural networks for price prediction with Proximal Policy Optimization (PPO) reinforcement learning for dynamic portfolio allocation.

## 🚀 Features

### Core Capabilities
- **LSTM Price Prediction**: Advanced multi-stock price forecasting using technical indicators
- **PPO Portfolio Optimization**: Reinforcement learning-based dynamic portfolio allocation
- **Risk Management**: Comprehensive risk profiling and constraint handling
- **Sector Analysis**: Detailed sector performance analysis and diversification strategies
- **Real-time Adaptation**: Dynamic rebalancing based on market conditions

### Key Components
1. **Multi-Stock LSTM Predictor** - Predicts future stock prices using technical indicators
2. **PPO Agent** - Learns optimal portfolio allocation strategies
3. **Risk-Aware Environment** - Incorporates user risk profiles and market constraints
4. **Sector Analyzer** - Provides sector-level insights and rotation strategies
5. **Portfolio Optimizer** - Main orchestrator combining all components

## 📁 Project Structure

```
portfolio_optimizer/
├── agents/
│   └── ppo_agent.py              # PPO agent implementation
├── environments/
│   └── portfolio_env.py          # Portfolio optimization environment
├── models/
│   └── lstm_predictor.py         # LSTM price prediction model
├── utils/
│   ├── data_loader.py            # Data loading and preprocessing
│   └── sector_analyzer.py        # Sector analysis utilities
├── portfolio_optimizer.py        # Main optimizer class
├── example_usage.py             # Usage examples
└── README.md                    # This file
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch
- scikit-learn
- pandas, numpy
- gym (for RL environment)

### Installation
```bash
# Install required packages
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy gym tqdm

# Optional: For enhanced features
pip install matplotlib seaborn plotly
```

### Data Requirements
1. **Stock Metadata**: `stocks.json` file with stock information
2. **Historical Data**: CSV file with historical price and technical indicators
3. **Feature Engineering**: Pre-calculated technical indicators (EMA, MACD, etc.)

## 🎯 Quick Start

### Basic Usage
```python
from portfolio_optimizer import PortfolioOptimizer

# Define risk profile
risk_profile = {
    'risk_tolerance': 'moderate',
    'investment_horizon': 'long',
    'preferred_sectors': ['Technology', 'Healthcare'],
    'excluded_sectors': ['Energy']
}

# Initialize optimizer
optimizer = PortfolioOptimizer(
    stocks_json_path="path/to/stocks.json",
    historical_data_path="path/to/historical_data.csv",
    risk_profile=risk_profile,
    initial_balance=100000.0
)

# Train the system
optimizer.train_system(
    lstm_epochs=100,
    ppo_episodes=500
)

# Get recommendations
recommendations = optimizer.get_portfolio_recommendations()
```

### Advanced Usage
```python
# Custom training parameters
optimizer.train_system(
    lstm_epochs=200,
    ppo_episodes=1000,
    save_models=True
)

# Get detailed insights
insights = optimizer.get_investment_insights()

# Sector analysis
from utils.sector_analyzer import create_sector_analyzer
sector_analyzer = create_sector_analyzer(stocks_path, historical_path)
sector_recs = sector_analyzer.get_top_sectors_recommendation()

# Save trained system
save_info = optimizer.save_system("my_portfolio_model")
```

## 🧠 System Architecture

### LSTM Price Predictor
- **Input**: Technical indicators (EMA, MACD, price data)
- **Architecture**: Multi-layer LSTM with attention mechanism
- **Output**: Price predictions for multiple stocks
- **Features**:
  - Bidirectional LSTM layers
  - Attention mechanism for better long-term dependencies
  - Dropout for regularization
  - Multi-stock training with individual scalers

### PPO Agent
- **State Space**: Portfolio weights, market indicators, LSTM predictions, risk metrics
- **Action Space**: Portfolio weight allocations (continuous)
- **Reward Function**: Risk-adjusted returns with diversification bonuses
- **Features**:
  - Actor-Critic architecture
  - Generalized Advantage Estimation (GAE)
  - Gradient clipping and early stopping
  - Experience buffer for efficient training

### Risk-Aware Environment
- **Risk Integration**: User risk profile constraints
- **Sector Preferences**: Preferred/excluded sector handling
- **Transaction Costs**: Realistic trading cost modeling
- **Diversification**: Automatic diversification scoring and bonuses

## 📊 Risk Profile Configuration

### Risk Tolerance Levels
```python
risk_profiles = {
    'conservative': {
        'target_volatility': 0.10,
        'max_drawdown_tolerance': 0.10,
        'preferred_sectors': ['Consumer Defensive', 'Utilities', 'Healthcare']
    },
    
    'moderate': {
        'target_volatility': 0.15,
        'max_drawdown_tolerance': 0.20,
        'preferred_sectors': []  # No specific preferences
    },
    
    'aggressive': {
        'target_volatility': 0.25,
        'max_drawdown_tolerance': 0.35,
        'preferred_sectors': ['Technology', 'Healthcare', 'Consumer Cyclical']
    }
}
```

### Custom Risk Profile
```python
custom_profile = {
    'risk_tolerance': 'moderate',
    'investment_horizon': 'long',
    'preferred_sectors': ['Technology', 'Healthcare'],
    'excluded_sectors': ['Energy', 'Basic Materials'],
    'target_volatility': 0.18,
    'max_drawdown_tolerance': 0.22,
    'min_dividend_yield': 0.02,
    'max_single_position': 0.15,
    'esg_filter': True
}
```

## 🎯 Output Examples

### Portfolio Recommendations
```json
{
  "timestamp": "2024-01-15",
  "portfolio_value": 100000.0,
  "recommended_allocations": {
    "AAPL": {
      "weight": 0.12,
      "dollar_amount": 12000,
      "sector": "Technology",
      "predicted_return": 0.08
    },
    "MSFT": {
      "weight": 0.10,
      "dollar_amount": 10000,
      "sector": "Technology",
      "predicted_return": 0.06
    }
  },
  "sector_allocations": {
    "Technology": 0.35,
    "Healthcare": 0.25,
    "Financial Services": 0.20
  },
  "risk_metrics": {
    "portfolio_volatility": 0.16,
    "portfolio_beta": 1.05,
    "sharpe_estimate": 1.2
  }
}
```

### Investment Insights
```json
{
  "market_analysis": {
    "market_trend": "neutral",
    "volatility_regime": "moderate"
  },
  "sector_analysis": {
    "top_sectors": ["Technology", "Healthcare", "Financial Services"],
    "sector_rotation_strategy": "growth_to_value"
  },
  "recommendations": [
    "Maintain current diversification levels",
    "Consider rebalancing monthly",
    "Monitor technology sector concentration"
  ]
}
```

## ⚡ Performance Optimization

### Training Tips
1. **Data Quality**: Ensure clean, consistent historical data
2. **Feature Engineering**: Use relevant technical indicators
3. **Hyperparameters**: Tune learning rates and network sizes
4. **Training Time**: Allow sufficient epochs for convergence

### Production Considerations
1. **Real-time Data**: Integrate with live market data feeds
2. **Model Updates**: Retrain models regularly with new data
3. **Risk Monitoring**: Implement real-time risk monitoring
4. **Execution**: Integrate with portfolio management systems

## 🔧 Customization

### Adding New Technical Indicators
```python
# In feature_engineering.py
def custom_indicator(df, window=20):
    return df['close'].rolling(window).apply(your_function)

# Add to feature preparation
def prepare_features(df):
    df['custom_indicator'] = custom_indicator(df)
    return df
```

### Custom Reward Function
```python
# In portfolio_env.py
def _calculate_custom_reward(self, portfolio_return, risk_metrics):
    base_reward = portfolio_return
    
    # Add custom components
    esg_bonus = self._calculate_esg_score() * 0.1
    momentum_bonus = self._calculate_momentum() * 0.05
    
    return base_reward + esg_bonus + momentum_bonus
```

### Sector Customization
```python
# Custom sector classifications
custom_sectors = {
    'ESG_Tech': ['AAPL', 'MSFT', 'GOOGL'],
    'Green_Energy': ['TSLA', 'ENPH', 'NEE']
}
```

## 📈 Monitoring & Evaluation

### Key Metrics
- **Return Metrics**: Total return, annualized return, Sharpe ratio
- **Risk Metrics**: Volatility, maximum drawdown, VaR
- **Diversification**: Sector allocation, concentration ratios
- **Transaction Costs**: Trading frequency, cost impact

### Performance Tracking
```python
# Evaluate system performance
evaluation = optimizer.evaluate_system()
print(f"Average Reward: {evaluation['average_reward']}")
print(f"Training Stability: {evaluation['training_stability']}")

# Get training statistics
stats = optimizer.ppo_agent.get_training_stats()
```

## 🚀 Advanced Features

### Dynamic Rebalancing
- Automatic rebalancing based on weight drift
- Transaction cost optimization
- Tax-loss harvesting integration

### Multi-Objective Optimization
- Simultaneous optimization of return, risk, and ESG scores
- Pareto-optimal portfolio frontier
- User preference integration

### Scenario Analysis
- Stress testing under different market conditions
- Monte Carlo simulations
- Sensitivity analysis

## 🔒 Risk Management

### Built-in Risk Controls
1. **Position Limits**: Maximum single position sizes
2. **Sector Limits**: Maximum sector allocations
3. **Volatility Control**: Target volatility constraints
4. **Drawdown Protection**: Maximum drawdown limits

### Risk Monitoring
```python
# Real-time risk monitoring
risk_metrics = optimizer.calculate_portfolio_risk_metrics(current_weights)
if risk_metrics['var_95'] < -0.05:  # 5% daily VaR threshold
    print("Risk limit exceeded - rebalancing recommended")
```

## 📝 Logging & Debugging

### Training Logs
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor training progress
optimizer.train_system(lstm_epochs=100, ppo_episodes=500)
```

### Performance Debugging
```python
# Analyze component performance
if lstm_success:
    print("LSTM predictions available")
else:
    print("Using historical returns for PPO training")

# Check environment observations
obs = optimizer.environment.reset()
print(f"Observation shape: {obs.shape}")
```

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Risk Management in Portfolio Optimization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1334674)

## ❓ FAQ

**Q: How long does training take?**
A: Training time depends on data size and complexity. Typical training takes 30-60 minutes on modern hardware.

**Q: Can I use this with any stock market?**
A: Yes, but you'll need appropriate historical data and stock metadata for your target market.

**Q: How often should I retrain the models?**
A: Recommended monthly retraining for LSTM, quarterly for PPO agent parameters.

**Q: What's the minimum amount of historical data needed?**
A: Minimum 2 years of daily data, but 5+ years recommended for robust training.

---

For questions and support, please open an issue in the repository.