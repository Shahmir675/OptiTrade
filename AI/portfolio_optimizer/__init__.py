"""
Portfolio Optimizer with LSTM + PPO
A comprehensive portfolio optimization system combining deep learning and reinforcement learning
"""

from .portfolio_optimizer import PortfolioOptimizer
from .models.lstm_predictor import LSTMPredictor
from .agents.ppo_agent import PPOAgent
from .environments.portfolio_env import RiskAwarePortfolioEnv
from .utils.data_loader import StockDataLoader
from .utils.sector_analyzer import SectorAnalyzer, create_sector_analyzer

__version__ = "1.0.0"
__author__ = "OptiTrade Development Team"

__all__ = [
    'PortfolioOptimizer',
    'LSTMPredictor', 
    'PPOAgent',
    'RiskAwarePortfolioEnv',
    'StockDataLoader',
    'SectorAnalyzer',
    'create_sector_analyzer'
]