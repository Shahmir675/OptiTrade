"""
Data loading and preprocessing utilities for Portfolio Optimizer
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
    """
    Loads and preprocesses stock data from various sources
    """
    
    def __init__(self, stocks_json_path: str, historical_csv_path: Optional[str] = None):
        self.stocks_json_path = stocks_json_path
        self.historical_csv_path = historical_csv_path
        self.stocks_info = None
        self.historical_data = None
        
    def load_stocks_metadata(self) -> Dict:
        """Load stocks metadata from JSON file"""
        stocks_data = []
        with open(self.stocks_json_path, 'r') as f:
            for line in f:
                stocks_data.append(json.loads(line.strip()))
        
        self.stocks_info = pd.DataFrame(stocks_data)
        return self.stocks_info
    
    def get_sector_distribution(self) -> Dict[str, int]:
        """Get distribution of stocks by sector"""
        if self.stocks_info is None:
            self.load_stocks_metadata()
        
        sector_dist = self.stocks_info['sector'].value_counts().to_dict()
        # Handle null sectors
        if pd.isna(list(sector_dist.keys())[0]):
            sector_dist['Unknown'] = sector_dist.pop(np.nan)
            
        return sector_dist
    
    def get_industry_distribution(self) -> Dict[str, int]:
        """Get distribution of stocks by industry"""
        if self.stocks_info is None:
            self.load_stocks_metadata()
            
        industry_dist = self.stocks_info['industry'].value_counts().to_dict()
        # Handle null industries
        if pd.isna(list(industry_dist.keys())[0]):
            industry_dist['Unknown'] = industry_dist.pop(np.nan)
            
        return industry_dist
    
    def filter_stocks_by_criteria(self, 
                                min_market_cap: float = 1e9,
                                min_volume: int = 100000,
                                exclude_sectors: List[str] = None) -> pd.DataFrame:
        """Filter stocks based on various criteria"""
        if self.stocks_info is None:
            self.load_stocks_metadata()
            
        filtered = self.stocks_info.copy()
        
        # Filter by market cap
        filtered = filtered[filtered['marketCap'] >= min_market_cap]
        
        # Filter by volume
        filtered = filtered[filtered['volume'] >= min_volume]
        
        # Exclude specific sectors
        if exclude_sectors:
            filtered = filtered[~filtered['sector'].isin(exclude_sectors)]
            
        return filtered
    
    def get_top_stocks_by_sector(self, n_per_sector: int = 10) -> Dict[str, List[str]]:
        """Get top N stocks per sector by market cap"""
        if self.stocks_info is None:
            self.load_stocks_metadata()
            
        top_stocks = {}
        for sector in self.stocks_info['sector'].unique():
            if pd.isna(sector):
                continue
                
            sector_stocks = self.stocks_info[self.stocks_info['sector'] == sector]
            sector_stocks = sector_stocks.nlargest(n_per_sector, 'marketCap')
            top_stocks[sector] = sector_stocks['symbol'].tolist()
            
        return top_stocks
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical price data"""
        if self.historical_csv_path:
            self.historical_data = pd.read_csv(self.historical_csv_path)
            self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
            return self.historical_data
        else:
            raise ValueError("Historical data path not provided")
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed info for a specific stock"""
        if self.stocks_info is None:
            self.load_stocks_metadata()
            
        stock_info = self.stocks_info[self.stocks_info['symbol'] == symbol]
        if not stock_info.empty:
            return stock_info.iloc[0].to_dict()
        else:
            return {}
    
    def get_diversified_portfolio_candidates(self, 
                                           target_stocks: int = 50,
                                           max_per_sector: int = 8) -> List[str]:
        """
        Get a diversified set of stock candidates for portfolio optimization
        """
        if self.stocks_info is None:
            self.load_stocks_metadata()
            
        # Filter out low-quality stocks
        quality_stocks = self.filter_stocks_by_criteria(
            min_market_cap=1e9,  # $1B market cap minimum
            min_volume=500000,   # 500K volume minimum
            exclude_sectors=['Shell Companies', None]
        )
        
        candidates = []
        sector_counts = {}
        
        # Sort by market cap to get highest quality stocks first
        quality_stocks = quality_stocks.sort_values('marketCap', ascending=False)
        
        for _, stock in quality_stocks.iterrows():
            sector = stock['sector']
            if pd.isna(sector):
                sector = 'Unknown'
                
            # Limit stocks per sector for diversification
            if sector_counts.get(sector, 0) < max_per_sector:
                candidates.append(stock['symbol'])
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                
                if len(candidates) >= target_stocks:
                    break
        
        return candidates[:target_stocks]


def load_feature_engineered_data(csv_path: str) -> pd.DataFrame:
    """Load the feature-engineered historical data"""
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def prepare_lstm_sequences(df: pd.DataFrame, 
                          sequence_length: int = 60,
                          prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training
    
    Args:
        df: DataFrame with features and target
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict ahead
        
    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples, prediction_horizon)
    """
    feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker']]
    
    X, y = [], []
    
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) < sequence_length + prediction_horizon:
            continue
            
        ticker_features = ticker_data[feature_cols].values
        
        for i in range(len(ticker_features) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(ticker_features[i:i + sequence_length])
            # Target (price prediction)
            y.append(ticker_features[i + sequence_length:i + sequence_length + prediction_horizon, 0])  # Assuming first column is price
    
    return np.array(X), np.array(y)