"""
Comprehensive Portfolio Optimizer
Combines LSTM price predictions with PPO reinforcement learning for optimal portfolio allocation
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.lstm_predictor import LSTMPredictor
from agents.ppo_agent import PPOAgent, PortfolioPPOTrainer
from environments.portfolio_env import RiskAwarePortfolioEnv
from utils.data_loader import StockDataLoader, load_feature_engineered_data


class PortfolioOptimizer:
    """
    Main portfolio optimization system combining LSTM predictions with PPO
    """
    
    def __init__(self,
                 stocks_json_path: str,
                 historical_data_path: str,
                 risk_profile: Dict[str, Any],
                 initial_balance: float = 100000.0,
                 max_stocks: int = 50,
                 rebalance_frequency: int = 5):
        
        self.stocks_json_path = stocks_json_path
        self.historical_data_path = historical_data_path
        self.risk_profile = risk_profile
        self.initial_balance = initial_balance
        self.max_stocks = max_stocks
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize components
        self.data_loader = StockDataLoader(stocks_json_path, historical_data_path)
        self.lstm_predictor = None
        self.ppo_agent = None
        self.environment = None
        self.trainer = None
        
        # Data storage
        self.stock_symbols = []
        self.historical_data = None
        self.current_portfolio = None
        self.optimization_results = {}
        
        # Load and prepare data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize and prepare all necessary data"""
        print("Initializing portfolio optimizer data...")
        
        # Load stock metadata
        stocks_info = self.data_loader.load_stocks_metadata()
        print(f"Loaded {len(stocks_info)} stocks from metadata")
        
        # Get diversified stock candidates
        self.stock_symbols = self.data_loader.get_diversified_portfolio_candidates(
            target_stocks=self.max_stocks
        )
        print(f"Selected {len(self.stock_symbols)} diversified stock candidates")
        
        # Load and update historical data
        try:
            # First, try to update historical data with fresh data from yfinance
            print("Updating historical data with fresh market data...")
            updated_data = self.data_loader.update_historical_data(
                symbols=self.stock_symbols, 
                save_path=self.historical_data_path
            )
            
            if not updated_data.empty:
                # Apply feature engineering to the updated data
                self.historical_data = load_feature_engineered_data(self.historical_data_path)
                
                # Filter for selected stocks
                self.historical_data = self.historical_data[
                    self.historical_data['Ticker'].isin(self.stock_symbols)
                ].copy()
                
                print(f"Loaded historical data with {len(self.historical_data)} records")
                print(f"Date range: {self.historical_data['Date'].min()} to {self.historical_data['Date'].max()}")
            else:
                raise ValueError("No historical data available")
            
        except Exception as e:
            print(f"Error loading/updating historical data: {e}")
            print("Falling back to fresh data fetch...")
            
            # Fallback: fetch fresh data directly
            try:
                fresh_data = self.data_loader.fetch_fresh_data(self.stock_symbols, period="5y")
                if not fresh_data.empty:
                    fresh_data.to_csv(self.historical_data_path, index=False)
                    self.historical_data = load_feature_engineered_data(self.historical_data_path)
                    
                    # Filter for selected stocks  
                    self.historical_data = self.historical_data[
                        self.historical_data['Ticker'].isin(self.stock_symbols)
                    ].copy()
                    
                    print(f"Fetched fresh data with {len(self.historical_data)} records")
                    print(f"Date range: {self.historical_data['Date'].min()} to {self.historical_data['Date'].max()}")
                else:
                    # Create dummy data for testing
                    self._create_dummy_data()
            except Exception as e2:
                print(f"Fresh data fetch also failed: {e2}")
                # Create dummy data for testing
                self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing purposes"""
        print("Creating dummy historical data for testing...")
        
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        data = []
        
        for symbol in self.stock_symbols:
            np.random.seed(hash(symbol) % 1000)  # Reproducible random data
            
            base_price = np.random.uniform(50, 200)
            prices = [base_price]
            
            for i in range(1, len(dates)):
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            for i, date in enumerate(dates):
                # Basic features
                price = prices[i]
                
                # Technical indicators (simplified)
                if i >= 20:
                    ema_20 = np.mean(prices[i-20:i])
                    ema_50 = np.mean(prices[i-50:i]) if i >= 50 else np.mean(prices[:i])
                else:
                    ema_20 = price
                    ema_50 = price
                
                data.append({
                    'Date': date,
                    'Ticker': symbol,
                    'Adj Close': price,
                    'EMA 20': ema_20,
                    'EMA 50': ema_50,
                    'MACD Line': (ema_20 - ema_50) / price,
                    'MACD Signal': (ema_20 - ema_50) / price * 0.9
                })
        
        self.historical_data = pd.DataFrame(data)
        print(f"Created dummy data with {len(self.historical_data)} records")
    
    def initialize_lstm_predictor(self, 
                                sequence_length: int = 60,
                                hidden_size: int = 128,
                                num_layers: int = 3,
                                epochs: int = 100):
        """Initialize and train LSTM predictor"""
        print("Initializing LSTM predictor...")
        
        # Ensure we have sufficient data before training
        min_required = sequence_length + 20  # Extra buffer for training
        if not self.ensure_sufficient_data(min_required):
            print(f"Warning: Unable to ensure sufficient data (need {min_required} days per stock)")
        
        self.lstm_predictor = LSTMPredictor(
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Prepare training data
        X, y, feature_cols = self.lstm_predictor.prepare_data(
            self.historical_data, target_col='Adj Close'
        )
        
        if len(X) == 0:
            print("Warning: No training data available for LSTM")
            return False
        
        print(f"Prepared {len(X)} training samples with {len(feature_cols)} features")
        
        # Train the model
        try:
            training_results = self.lstm_predictor.train(
                X, y, epochs=epochs, batch_size=32
            )
            
            print("LSTM training completed successfully")
            print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"LSTM training failed: {e}")
            return False

    def ensure_sufficient_data(self, min_days_required: int = 120) -> bool:
        """
        Ensure we have sufficient historical data for LSTM training.
        If not, fetch more data from yfinance.
        
        Args:
            min_days_required: Minimum number of days of data required per stock
        """
        if self.historical_data is None or self.historical_data.empty:
            print("No historical data available, fetching fresh data...")
            try:
                fresh_data = self.data_loader.fetch_fresh_data(self.stock_symbols, period="5y")
                if not fresh_data.empty:
                    fresh_data.to_csv(self.historical_data_path, index=False)
                    from utils.data_loader import load_feature_engineered_data
                    self.historical_data = load_feature_engineered_data(self.historical_data_path)
                    return True
            except Exception as e:
                print(f"Failed to fetch fresh data: {e}")
                return False
        
        # Check data sufficiency per ticker
        ticker_counts = self.historical_data['Ticker'].value_counts()
        insufficient_tickers = ticker_counts[ticker_counts < min_days_required]
        
        if len(insufficient_tickers) == 0:
            print("All tickers have sufficient data")
            return True
            
        print(f"Found {len(insufficient_tickers)} tickers with insufficient data (< {min_days_required} days)")
        print(f"Insufficient tickers: {insufficient_tickers.index[:5].tolist()}...")
        
        # Fetch more historical data for insufficient tickers
        try:
            print("Fetching extended historical data...")
            extended_data = self.data_loader.fetch_fresh_data(
                symbols=list(insufficient_tickers.index), 
                period="5y"  # Get 5 years of data
            )
            
            if not extended_data.empty:
                # Combine with existing data
                combined = pd.concat([self.historical_data, extended_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
                combined = combined.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                
                # Save and reload with feature engineering
                combined.to_csv(self.historical_data_path, index=False)
                from utils.data_loader import load_feature_engineered_data
                self.historical_data = load_feature_engineered_data(self.historical_data_path)
                
                # Filter for selected stocks
                self.historical_data = self.historical_data[
                    self.historical_data['Ticker'].isin(self.stock_symbols)
                ].copy()
                
                print(f"Extended data loaded: {len(self.historical_data)} records")
                return True
            else:
                print("No extended data could be fetched")
                return False
                
        except Exception as e:
            print(f"Failed to fetch extended data: {e}")
            return False
    
    def initialize_ppo_agent(self, 
                           learning_rate: float = 3e-4,
                           gamma: float = 0.99,
                           eps_clip: float = 0.2):
        """Initialize PPO agent and environment"""
        print("Initializing PPO agent and environment...")
        
        # Create environment
        self.environment = RiskAwarePortfolioEnv(
            risk_profile=self.risk_profile,
            stock_data=self.historical_data,
            lstm_predictor=self.lstm_predictor,
            stock_symbols=self.stock_symbols,
            initial_balance=self.initial_balance,
            rebalance_frequency=self.rebalance_frequency
        )
        
        # Get environment dimensions
        state_dim = self.environment.observation_space.shape[0]
        action_dim = self.environment.action_space.shape[0]
        
        print(f"Environment: {state_dim} state dims, {action_dim} action dims")
        
        # Create PPO agent
        self.ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=learning_rate,
            lr_critic=learning_rate,
            gamma=gamma,
            eps_clip=eps_clip
        )
        
        # Create trainer (will be updated with episodes during training)
        self.trainer = PortfolioPPOTrainer(
            env=self.environment,
            agent=self.ppo_agent
        )
        
        print("PPO agent initialized successfully")
        return True
    
    def train_system(self, 
                   lstm_epochs: int = 100,
                   ppo_episodes: int = 500,
                   save_models: bool = True):
        """Train the complete system (LSTM + PPO)"""
        print("Starting comprehensive system training...")
        print(f"Target LSTM epochs: {lstm_epochs}")
        print(f"Target PPO episodes: {ppo_episodes}")
        
        # Step 1: Train LSTM predictor
        print("\n=== Phase 1: Training LSTM Predictor ===")
        print("Status: Initializing LSTM predictor...")
        lstm_success = self.initialize_lstm_predictor(epochs=lstm_epochs)
        
        if not lstm_success:
            print("❌ LSTM training failed. Continuing with PPO training...")
        else:
            print("✅ LSTM training completed successfully")
        
        # Step 2: Initialize and train PPO agent
        print("\n=== Phase 2: Training PPO Agent ===")
        print("Status: Initializing PPO agent and environment...")
        ppo_success = self.initialize_ppo_agent()
        
        if not ppo_success:
            print("❌ PPO initialization failed")
            return False
        
        print("✅ PPO agent initialized successfully")
        
        # Train PPO
        print(f"Status: Starting PPO training for {ppo_episodes} episodes...")
        save_path = "portfolio_ppo_model" if save_models else None
        self.trainer.max_episodes = ppo_episodes  # Set episodes for this training run
        training_results = self.trainer.train(save_path=save_path)
        
        self.optimization_results = {
            'lstm_trained': lstm_success,
            'ppo_training_results': training_results,
            'final_evaluation': self.evaluate_system()
        }
        
        print("\n=== 🎯 Training Complete ===")
        print(f"LSTM Training: {'✅ Success' if lstm_success else '❌ Failed'}")
        print(f"PPO Episodes Completed: {len(training_results.get('episode_rewards', []))}/{ppo_episodes}")
        
        if training_results.get('episode_rewards'):
            avg_final_reward = np.mean(training_results['episode_rewards'][-10:])
            max_reward = max(training_results['episode_rewards'])
            print(f"📈 Final Average Reward (last 10 episodes): {avg_final_reward:.4f}")
            print(f"🏆 Best Episode Reward: {max_reward:.4f}")
            print(f"📊 Training Progress: {len(training_results['episode_rewards'])} episodes completed")
        else:
            print("⚠️  No training rewards recorded")
        
        # System evaluation
        evaluation = self.evaluate_system()
        if evaluation:
            print(f"🔍 System Evaluation Score: {evaluation.get('total_score', 'N/A')}")
        
        print(f"💾 Models Saved: {'Yes' if save_models else 'No'}")
        
        return True
    
    def get_portfolio_recommendations(self, 
                                    current_date: Optional[str] = None,
                                    portfolio_value: Optional[float] = None) -> Dict[str, Any]:
        """Get current portfolio recommendations"""
        if self.ppo_agent is None:
            raise ValueError("System must be trained before getting recommendations")
        
        if current_date is None:
            current_date = self.historical_data['Date'].max()
        else:
            current_date = pd.to_datetime(current_date)
        
        if portfolio_value is None:
            portfolio_value = self.initial_balance
        
        # Get current market state
        current_state = self._get_current_market_state(current_date, portfolio_value)
        
        # Get PPO recommendation
        recommended_weights, _ = self.ppo_agent.get_action(current_state, deterministic=True)
        
        # Get LSTM price predictions
        price_predictions = {}
        if self.lstm_predictor is not None:
            try:
                recent_data = self.historical_data[
                    self.historical_data['Date'] <= current_date
                ].tail(1000)  # Get recent data for predictions
                
                price_predictions = self.lstm_predictor.predict_stock_prices(
                    recent_data, self.stock_symbols
                )
            except Exception as e:
                print(f"Price prediction failed: {e}")
        
        # Create recommendations
        recommendations = {
            'timestamp': current_date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'recommended_allocations': {},
            'sector_allocations': {},
            'price_predictions': price_predictions,
            'risk_metrics': self._calculate_portfolio_risk_metrics(recommended_weights),
            'rebalancing_needed': self._assess_rebalancing_need(recommended_weights)
        }
        
        # Stock allocations
        for i, symbol in enumerate(self.stock_symbols):
            if recommended_weights[i] > 0.01:  # Only include significant allocations
                stock_info = self.data_loader.get_stock_info(symbol)
                
                recommendations['recommended_allocations'][symbol] = {
                    'weight': float(recommended_weights[i]),
                    'dollar_amount': float(recommended_weights[i] * portfolio_value),
                    'sector': stock_info.get('sector', 'Unknown'),
                    'industry': stock_info.get('industry', 'Unknown'),
                    'current_price': stock_info.get('close', 0),
                    'predicted_return': price_predictions.get(symbol, 0) if price_predictions else 0
                }
        
        # Sector allocations
        sector_weights = {}
        for symbol, allocation in recommendations['recommended_allocations'].items():
            sector = allocation['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + allocation['weight']
        
        recommendations['sector_allocations'] = sector_weights
        
        return recommendations
    
    def get_investment_insights(self) -> Dict[str, Any]:
        """Get comprehensive investment insights and analysis"""
        insights = {
            'market_analysis': self._analyze_market_conditions(),
            'sector_analysis': self._analyze_sector_opportunities(),
            'risk_analysis': self._analyze_portfolio_risk(),
            'performance_attribution': self._analyze_performance_attribution(),
            'recommendations': self._generate_investment_recommendations()
        }
        
        return insights
    
    def _get_current_market_state(self, date, portfolio_value) -> np.ndarray:
        """Get current market state for PPO agent"""
        if self.environment is None:
            # Create minimal state
            return np.zeros(100)  # Placeholder
        
        # Reset environment to get current state
        try:
            # Filter data up to current date
            historical_subset = self.historical_data[
                self.historical_data['Date'] <= date
            ].copy()
            
            # Update environment data
            self.environment.stock_data = historical_subset
            
            # Get observation
            state = self.environment.reset()
            return state
            
        except Exception as e:
            print(f"Error getting market state: {e}")
            # Return zero state as fallback
            return np.zeros(self.environment.observation_space.shape[0])
    
    def _calculate_portfolio_risk_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics for given portfolio weights"""
        if len(weights) == 0 or np.sum(weights) == 0:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_beta': 0.0,
                'max_drawdown_estimate': 0.0,
                'var_95': 0.0,
                'sharpe_estimate': 0.0
            }
        
        # Get recent returns data
        returns_data = []
        for symbol in self.stock_symbols:
            symbol_data = self.historical_data[
                self.historical_data['Ticker'] == symbol
            ].sort_values('Date').tail(252)  # Last year
            
            if len(symbol_data) > 1:
                prices = symbol_data['Adj Close'].values
                returns = np.diff(prices) / prices[:-1]
                returns_data.append(returns)
        
        if not returns_data:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_beta': 0.0,
                'max_drawdown_estimate': 0.0,
                'var_95': 0.0,
                'sharpe_estimate': 0.0
            }
        
        # Align returns data
        min_length = min(len(returns) for returns in returns_data)
        aligned_returns = np.array([returns[-min_length:] for returns in returns_data])
        
        # Portfolio returns
        portfolio_returns = np.sum(aligned_returns * weights.reshape(-1, 1), axis=0)
        
        # Risk metrics
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Beta (vs equal-weighted market)
        market_returns = np.mean(aligned_returns, axis=0)
        portfolio_beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 0
        
        # VaR
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Sharpe ratio estimate
        mean_return = np.mean(portfolio_returns) * 252
        sharpe_estimate = mean_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Max drawdown estimate
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown_estimate = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'portfolio_volatility': float(portfolio_volatility),
            'portfolio_beta': float(portfolio_beta),
            'max_drawdown_estimate': float(max_drawdown_estimate),
            'var_95': float(var_95),
            'sharpe_estimate': float(sharpe_estimate)
        }
    
    def _assess_rebalancing_need(self, new_weights: np.ndarray) -> Dict[str, Any]:
        """Assess if rebalancing is needed"""
        if self.current_portfolio is None:
            return {
                'rebalancing_needed': True,
                'reason': 'Initial portfolio allocation',
                'weight_changes': {}
            }
        
        # Calculate weight changes
        weight_changes = {}
        total_change = 0
        
        for i, symbol in enumerate(self.stock_symbols):
            old_weight = self.current_portfolio.get(symbol, 0)
            new_weight = new_weights[i] if i < len(new_weights) else 0
            change = abs(new_weight - old_weight)
            
            if change > 0.01:  # 1% threshold
                weight_changes[symbol] = {
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'change': change
                }
            
            total_change += change
        
        rebalancing_needed = total_change > 0.05  # 5% total change threshold
        
        return {
            'rebalancing_needed': rebalancing_needed,
            'total_weight_change': total_change,
            'significant_changes': weight_changes,
            'reason': 'Significant weight changes detected' if rebalancing_needed else 'Portfolio weights stable'
        }
    
    def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        # This would integrate with real market data
        # For now, return basic analysis
        return {
            'market_trend': 'neutral',
            'volatility_regime': 'moderate',
            'sector_rotation': 'technology_to_value',
            'economic_indicators': {
                'interest_rate_environment': 'rising',
                'inflation_trend': 'moderating',
                'economic_growth': 'stable'
            }
        }
    
    def _analyze_sector_opportunities(self) -> Dict[str, Any]:
        """Analyze sector-specific opportunities"""
        # Get sector performance from historical data
        sector_performance = {}
        
        if self.data_loader.stocks_info is not None:
            sector_dist = self.data_loader.get_sector_distribution()
            
            for sector, count in sector_dist.items():
                if sector != 'Unknown':
                    sector_performance[sector] = {
                        'stock_count': count,
                        'recommendation': 'neutral',  # Would be calculated
                        'expected_return': 0.0,      # Would be calculated
                        'risk_level': 'moderate'     # Would be calculated
                    }
        
        return {
            'sector_performance': sector_performance,
            'top_sectors': list(sector_performance.keys())[:5],
            'sector_rotation_strategy': 'diversified_approach'
        }
    
    def _analyze_portfolio_risk(self) -> Dict[str, Any]:
        """Analyze portfolio risk characteristics"""
        return {
            'risk_level': self.risk_profile.get('risk_tolerance', 'moderate'),
            'diversification_score': 0.8,  # Would be calculated
            'concentration_risk': 'low',
            'sector_concentration': 'well_diversified',
            'recommendations': [
                'Maintain current diversification levels',
                'Consider rebalancing monthly',
                'Monitor sector allocation limits'
            ]
        }
    
    def _analyze_performance_attribution(self) -> Dict[str, Any]:
        """Analyze performance attribution"""
        return {
            'asset_allocation_contribution': 0.0,
            'security_selection_contribution': 0.0,
            'interaction_effect': 0.0,
            'total_active_return': 0.0,
            'benchmark_comparison': 'market_neutral'
        }
    
    def _generate_investment_recommendations(self) -> List[str]:
        """Generate specific investment recommendations"""
        recommendations = []
        
        risk_tolerance = self.risk_profile.get('risk_tolerance', 'moderate')
        
        if risk_tolerance == 'conservative':
            recommendations.extend([
                'Focus on dividend-paying stocks and defensive sectors',
                'Maintain higher cash allocation for stability',
                'Consider bond allocation for income generation'
            ])
        elif risk_tolerance == 'aggressive':
            recommendations.extend([
                'Consider growth stocks and emerging sectors',
                'Utilize leverage opportunities when appropriate',
                'Focus on momentum strategies'
            ])
        else:  # moderate
            recommendations.extend([
                'Maintain balanced allocation across sectors',
                'Regular rebalancing to maintain target weights',
                'Consider both growth and value opportunities'
            ])
        
        return recommendations
    
    def evaluate_system(self) -> Dict[str, float]:
        """Evaluate the complete system performance"""
        if self.trainer is None:
            return {'error': 'System not trained'}
        
        # Run evaluation
        eval_reward = self.trainer.evaluate_agent(num_episodes=10)
        
        # Get additional metrics
        training_stats = self.ppo_agent.get_training_stats() if self.ppo_agent else {}
        
        return {
            'average_reward': eval_reward,
            'training_stability': np.std(training_stats.get('actor_loss', [0])),
            'convergence_score': 1.0 / (1.0 + np.std(training_stats.get('actor_loss', [1]))),
            'system_status': 'trained' if self.ppo_agent else 'not_trained'
        }
    
    def save_system(self, base_path: str):
        """Save the complete trained system"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save LSTM model
        if self.lstm_predictor is not None:
            lstm_path = f"{base_path}_lstm_{timestamp}.pth"
            self.lstm_predictor.save_model(lstm_path)
        
        # Save PPO model
        if self.ppo_agent is not None:
            ppo_path = f"{base_path}_ppo_{timestamp}.pth"
            self.ppo_agent.save_model(ppo_path)
        
        # Save system configuration
        config = {
            'stock_symbols': self.stock_symbols,
            'risk_profile': self.risk_profile,
            'initial_balance': self.initial_balance,
            'max_stocks': self.max_stocks,
            'optimization_results': self.optimization_results,
            'timestamp': timestamp
        }
        
        config_path = f"{base_path}_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"System saved with timestamp: {timestamp}")
        return {
            'lstm_model': lstm_path if self.lstm_predictor else None,
            'ppo_model': ppo_path if self.ppo_agent else None,
            'config': config_path,
            'timestamp': timestamp
        }
    
    def load_system(self, config_path: str):
        """Load a previously saved system"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Restore configuration
        self.stock_symbols = config['stock_symbols']
        self.risk_profile = config['risk_profile']
        self.initial_balance = config['initial_balance']
        self.max_stocks = config['max_stocks']
        self.optimization_results = config.get('optimization_results', {})
        
        print(f"System configuration loaded from {config_path}")
        return config