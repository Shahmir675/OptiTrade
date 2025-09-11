"""
Portfolio Optimization Environment for PPO Agent
Includes risk management, sector diversification, and LSTM price predictions
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizationEnv(gym.Env):
    """
    Environment for portfolio optimization using reinforcement learning
    """
    
    def __init__(self,
                 stock_data: pd.DataFrame,
                 lstm_predictor,
                 stock_symbols: List[str],
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.2,
                 risk_free_rate: float = 0.02,
                 lookback_window: int = 60,
                 rebalance_frequency: int = 5):
        
        super(PortfolioOptimizationEnv, self).__init__()
        
        # Core parameters
        self.stock_data = stock_data
        self.lstm_predictor = lstm_predictor
        self.stock_symbols = stock_symbols
        self.n_stocks = len(stock_symbols)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        
        # Environment state
        self.current_step = 0
        self.current_portfolio = np.zeros(self.n_stocks)  # Portfolio weights
        self.cash_balance = initial_balance
        self.total_value = initial_balance
        self.transaction_costs_incurred = 0.0
        
        # Track performance metrics
        self.portfolio_values = [initial_balance]
        self.returns = []
        self.volatility_window = []
        self.drawdown_history = []
        self.sector_allocations = []
        
        # Get sector information
        self.stock_sectors = self._get_stock_sectors()
        
        # Define action and observation spaces
        # Action space: weights for each stock (sum should be <= 1, remaining is cash)
        self.action_space = spaces.Box(
            low=0.0, 
            high=self.max_position_size, 
            shape=(self.n_stocks,), 
            dtype=np.float32
        )
        
        # Observation space: 
        # - Current portfolio weights (n_stocks)
        # - Recent returns for each stock (n_stocks * lookback_window)
        # - LSTM predictions (n_stocks)
        # - Portfolio metrics (5): total_return, volatility, sharpe_ratio, max_drawdown, cash_ratio
        # - Sector diversification score (1)
        # - Risk metrics (3): beta, var, expected_shortfall
        obs_size = (self.n_stocks +  # current weights
                   self.n_stocks * self.lookback_window +  # historical returns
                   self.n_stocks +  # LSTM predictions  
                   5 +  # portfolio metrics
                   1 +  # sector diversification
                   3)   # risk metrics
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initialize data
        self.returns_data = self._calculate_returns()
        self.price_data = self._get_price_data()
        
    def _get_stock_sectors(self) -> Dict[str, str]:
        """Map each stock to its sector"""
        sectors = {}
        # This would be populated from the stock metadata
        # For now, using dummy sectors - should be integrated with actual data
        sector_list = ['Technology', 'Healthcare', 'Financial Services', 
                      'Consumer Cyclical', 'Industrials', 'Energy', 
                      'Basic Materials', 'Consumer Defensive']
        
        for i, symbol in enumerate(self.stock_symbols):
            sectors[symbol] = sector_list[i % len(sector_list)]
        
        return sectors
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all stocks"""
        returns_data = {}
        
        for symbol in self.stock_symbols:
            symbol_data = self.stock_data[self.stock_data['Ticker'] == symbol].sort_values('Date')
            if len(symbol_data) > 0:
                prices = symbol_data['Adj Close'].values
                returns = np.diff(prices) / prices[:-1]
                returns_data[symbol] = pd.Series(returns, index=symbol_data['Date'].iloc[1:])
        
        return pd.DataFrame(returns_data)
    
    def _get_price_data(self) -> pd.DataFrame:
        """Get price data for all stocks"""
        price_data = {}
        
        for symbol in self.stock_symbols:
            symbol_data = self.stock_data[self.stock_data['Ticker'] == symbol].sort_values('Date')
            if len(symbol_data) > 0:
                price_data[symbol] = pd.Series(
                    symbol_data['Adj Close'].values,
                    index=symbol_data['Date']
                )
        
        return pd.DataFrame(price_data)
    
    def _get_lstm_predictions(self, current_date) -> np.ndarray:
        """Get LSTM predictions for all stocks"""
        predictions = np.zeros(self.n_stocks)
        
        if self.lstm_predictor is not None:
            try:
                # Get recent data up to current date
                recent_data = self.stock_data[
                    self.stock_data['Date'] <= current_date
                ].tail(self.lookback_window * len(self.stock_symbols))
                
                lstm_preds = self.lstm_predictor.predict_stock_prices(
                    recent_data, self.stock_symbols
                )
                
                for i, symbol in enumerate(self.stock_symbols):
                    if symbol in lstm_preds:
                        current_price = self._get_current_price(symbol, current_date)
                        if current_price > 0:
                            # Convert to expected return
                            predictions[i] = (lstm_preds[symbol] - current_price) / current_price
            except Exception as e:
                print(f"Warning: LSTM prediction failed: {e}")
                predictions = np.zeros(self.n_stocks)
        
        return predictions
    
    def _get_current_price(self, symbol: str, date) -> float:
        """Get current price for a stock"""
        symbol_data = self.stock_data[
            (self.stock_data['Ticker'] == symbol) & 
            (self.stock_data['Date'] <= date)
        ].sort_values('Date')
        
        if len(symbol_data) > 0:
            return float(symbol_data['Adj Close'].iloc[-1])
        return 0.0
    
    def _get_current_returns(self, date, window: int = None) -> np.ndarray:
        """Get recent returns for all stocks"""
        if window is None:
            window = self.lookback_window
            
        returns = np.zeros((self.n_stocks, window))
        
        for i, symbol in enumerate(self.stock_symbols):
            if symbol in self.returns_data.columns:
                symbol_returns = self.returns_data[symbol][
                    self.returns_data.index <= date
                ].tail(window).values
                
                if len(symbol_returns) == window:
                    returns[i, :] = symbol_returns
        
        return returns
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate current portfolio performance metrics"""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'cash_ratio': 1.0
            }
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Total return
        total_return = (values[-1] - values[0]) / values[0]
        
        # Volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = 0.0
        
        # Sharpe ratio
        if volatility > 0:
            excess_return = np.mean(returns) * 252 - self.risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Cash ratio
        cash_ratio = self.cash_balance / self.total_value
        
        return {
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'cash_ratio': float(cash_ratio)
        }
    
    def _calculate_sector_diversification(self) -> float:
        """Calculate sector diversification score"""
        if np.sum(self.current_portfolio) == 0:
            return 1.0
        
        sector_weights = {}
        for i, symbol in enumerate(self.stock_symbols):
            sector = self.stock_sectors.get(symbol, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + self.current_portfolio[i]
        
        # Calculate Herfindahl-Hirschman Index (lower is better for diversification)
        hhi = sum(weight**2 for weight in sector_weights.values())
        
        # Convert to diversification score (higher is better)
        max_hhi = 1.0  # Maximum concentration (all in one sector)
        diversification_score = 1.0 - (hhi / max_hhi)
        
        return float(diversification_score)
    
    def _calculate_risk_metrics(self, current_date) -> Dict[str, float]:
        """Calculate risk metrics for current portfolio"""
        returns_window = self._get_current_returns(current_date, window=30)
        
        if returns_window.size == 0 or np.sum(self.current_portfolio) == 0:
            return {'beta': 0.0, 'var': 0.0, 'expected_shortfall': 0.0}
        
        # Portfolio returns
        portfolio_returns = np.sum(returns_window * self.current_portfolio.reshape(-1, 1), axis=0)
        
        # Beta (assuming market return is average of all stock returns)
        market_returns = np.mean(returns_window, axis=0)
        if np.var(market_returns) > 0:
            beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
        else:
            beta = 0.0
        
        # Value at Risk (5% VaR)
        if len(portfolio_returns) > 0:
            var = np.percentile(portfolio_returns, 5)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = portfolio_returns[portfolio_returns <= var]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var
        else:
            var = 0.0
            expected_shortfall = 0.0
        
        return {
            'beta': float(beta),
            'var': float(var),
            'expected_shortfall': float(expected_shortfall)
        }
    
    def _get_observation(self, current_date) -> np.ndarray:
        """Get current observation state"""
        obs = []
        
        # Current portfolio weights
        obs.extend(self.current_portfolio)
        
        # Recent returns (flattened)
        recent_returns = self._get_current_returns(current_date)
        obs.extend(recent_returns.flatten())
        
        # LSTM predictions
        lstm_preds = self._get_lstm_predictions(current_date)
        obs.extend(lstm_preds)
        
        # Portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        obs.extend([
            portfolio_metrics['total_return'],
            portfolio_metrics['volatility'],
            portfolio_metrics['sharpe_ratio'],
            portfolio_metrics['max_drawdown'],
            portfolio_metrics['cash_ratio']
        ])
        
        # Sector diversification
        sector_div = self._calculate_sector_diversification()
        obs.append(sector_div)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(current_date)
        obs.extend([
            risk_metrics['beta'],
            risk_metrics['var'],
            risk_metrics['expected_shortfall']
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, current_date, new_weights: np.ndarray) -> float:
        """Calculate reward for current action"""
        # Get current prices
        current_prices = np.array([
            self._get_current_price(symbol, current_date) 
            for symbol in self.stock_symbols
        ])
        
        # Calculate portfolio value
        portfolio_value = np.sum(self.current_portfolio * current_prices) + self.cash_balance
        previous_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        
        # Returns-based reward
        portfolio_return = (portfolio_value - previous_value) / previous_value
        
        # Risk-adjusted reward components
        portfolio_metrics = self._calculate_portfolio_metrics()
        
        # Sharpe ratio bonus
        sharpe_bonus = max(0, portfolio_metrics['sharpe_ratio']) * 0.1
        
        # Diversification bonus
        div_score = self._calculate_sector_diversification()
        diversification_bonus = div_score * 0.05
        
        # Transaction cost penalty
        weight_changes = np.abs(new_weights - self.current_portfolio)
        transaction_penalty = np.sum(weight_changes) * self.transaction_cost * 10
        
        # Risk penalty (penalize high drawdown)
        drawdown_penalty = abs(portfolio_metrics['max_drawdown']) * 0.5
        
        # Concentration penalty (penalize positions > max_position_size)
        concentration_penalty = np.sum(np.maximum(0, new_weights - self.max_position_size)) * 2.0
        
        # Total reward
        reward = (portfolio_return + 
                 sharpe_bonus + 
                 diversification_bonus - 
                 transaction_penalty - 
                 drawdown_penalty - 
                 concentration_penalty)
        
        return float(reward)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment"""
        # Normalize action (ensure weights sum to <= 1)
        action = np.clip(action, 0, self.max_position_size)
        total_weight = np.sum(action)
        
        if total_weight > 1.0:
            action = action / total_weight
        
        # Get current date
        dates = sorted(self.stock_data['Date'].unique())
        if self.current_step >= len(dates):
            return self._get_observation(dates[-1]), 0.0, True, {'reason': 'end_of_data'}
        
        current_date = dates[self.current_step]
        
        # Calculate reward
        reward = self._calculate_reward(current_date, action)
        
        # Update portfolio
        previous_weights = self.current_portfolio.copy()
        self.current_portfolio = action
        
        # Calculate transaction costs
        weight_changes = np.abs(action - previous_weights)
        transaction_cost = np.sum(weight_changes) * self.transaction_cost * self.total_value
        self.transaction_costs_incurred += transaction_cost
        
        # Update portfolio value
        current_prices = np.array([
            self._get_current_price(symbol, current_date) 
            for symbol in self.stock_symbols
        ])
        
        portfolio_value = np.sum(self.current_portfolio * current_prices) - transaction_cost
        self.cash_balance = self.total_value * (1 - np.sum(self.current_portfolio))
        self.total_value = portfolio_value + self.cash_balance
        
        # Update tracking
        self.portfolio_values.append(self.total_value)
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= len(dates) - 1
        
        # Get new observation
        next_obs = self._get_observation(current_date)
        
        # Info dictionary
        info = {
            'portfolio_value': self.total_value,
            'cash_balance': self.cash_balance,
            'transaction_costs': self.transaction_costs_incurred,
            'portfolio_weights': self.current_portfolio.tolist(),
            'reward_components': {
                'portfolio_return': (self.total_value - self.portfolio_values[-2]) / self.portfolio_values[-2] if len(self.portfolio_values) > 1 else 0,
                'transaction_cost': transaction_cost,
                'diversification_score': self._calculate_sector_diversification()
            }
        }
        
        return next_obs, reward, done, info
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.current_step = 0
        self.current_portfolio = np.zeros(self.n_stocks)
        self.cash_balance = self.initial_balance
        self.total_value = self.initial_balance
        self.transaction_costs_incurred = 0.0
        
        # Reset tracking
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        self.volatility_window = []
        self.drawdown_history = []
        
        # Get initial observation
        dates = sorted(self.stock_data['Date'].unique())
        if len(dates) > 0:
            initial_obs = self._get_observation(dates[0])
        else:
            # Fallback observation
            initial_obs = np.zeros(self.observation_space.shape[0])
        
        return initial_obs
    
    def render(self, mode: str = 'human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.total_value:,.2f}")
            print(f"Cash Balance: ${self.cash_balance:,.2f}")
            print(f"Portfolio Weights: {self.current_portfolio}")
            print(f"Transaction Costs: ${self.transaction_costs_incurred:,.2f}")
            print("-" * 50)


class RiskAwarePortfolioEnv(PortfolioOptimizationEnv):
    """
    Extended environment with user risk profile integration
    """
    
    def __init__(self, risk_profile: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.risk_profile = risk_profile
        self.risk_tolerance = risk_profile.get('risk_tolerance', 'moderate')  # conservative, moderate, aggressive
        self.investment_horizon = risk_profile.get('investment_horizon', 'medium')  # short, medium, long
        self.preferred_sectors = risk_profile.get('preferred_sectors', [])
        self.excluded_sectors = risk_profile.get('excluded_sectors', [])
        
        # Adjust parameters based on risk profile
        self._adjust_parameters_for_risk_profile()
    
    def _adjust_parameters_for_risk_profile(self):
        """Adjust environment parameters based on user risk profile"""
        if self.risk_tolerance == 'conservative':
            self.max_position_size = min(self.max_position_size, 0.15)
            self.transaction_cost *= 1.2  # More conservative about trading
        elif self.risk_tolerance == 'aggressive':
            self.max_position_size = min(self.max_position_size, 0.3)
            self.transaction_cost *= 0.8  # Less concern about transaction costs
    
    def _calculate_risk_adjusted_reward(self, base_reward: float, portfolio_metrics: Dict) -> float:
        """Adjust reward based on user risk profile"""
        adjusted_reward = base_reward
        
        # Risk tolerance adjustments
        if self.risk_tolerance == 'conservative':
            # Penalize high volatility more
            volatility_penalty = portfolio_metrics.get('volatility', 0) * 0.5
            adjusted_reward -= volatility_penalty
            
            # Bonus for low drawdown
            drawdown_bonus = max(0, -portfolio_metrics.get('max_drawdown', 0)) * 0.3
            adjusted_reward += drawdown_bonus
            
        elif self.risk_tolerance == 'aggressive':
            # Bonus for higher returns, less penalty for volatility
            return_bonus = portfolio_metrics.get('total_return', 0) * 0.2
            adjusted_reward += return_bonus
        
        return adjusted_reward
    
    def _apply_sector_preferences(self, action: np.ndarray) -> np.ndarray:
        """Apply sector preferences to action"""
        adjusted_action = action.copy()
        
        for i, symbol in enumerate(self.stock_symbols):
            sector = self.stock_sectors.get(symbol, 'Unknown')
            
            # Reduce weight for excluded sectors
            if sector in self.excluded_sectors:
                adjusted_action[i] *= 0.5
            
            # Increase weight for preferred sectors
            if sector in self.preferred_sectors:
                adjusted_action[i] *= 1.2
        
        # Renormalize
        total_weight = np.sum(adjusted_action)
        if total_weight > 1.0:
            adjusted_action = adjusted_action / total_weight
        
        return adjusted_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step with risk profile considerations"""
        # Apply sector preferences
        risk_adjusted_action = self._apply_sector_preferences(action)
        
        # Call parent step method
        obs, base_reward, done, info = super().step(risk_adjusted_action)
        
        # Adjust reward based on risk profile
        portfolio_metrics = self._calculate_portfolio_metrics()
        adjusted_reward = self._calculate_risk_adjusted_reward(base_reward, portfolio_metrics)
        
        # Add risk profile info
        info['risk_profile'] = self.risk_profile
        info['risk_adjusted_reward'] = adjusted_reward
        
        return obs, adjusted_reward, done, info