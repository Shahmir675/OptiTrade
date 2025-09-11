"""
Data loading and preprocessing utilities for Portfolio Optimizer
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import yfinance as yf
import curl_cffi.requests
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
        
        # Setup proxy and session for yfinance
        self._setup_yfinance_proxy()
        
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
    
    def fetch_fresh_data(self, symbols: List[str], period: str = "2y", interval: str = "1d", 
                          batch_size: int = 50, max_workers: int = 10) -> pd.DataFrame:
        """
        Fetch fresh historical data from yfinance for given symbols with parallel processing
        
        Args:
            symbols: List of stock symbols to fetch
            period: Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
            batch_size: Number of symbols to fetch at once using yfinance download
            max_workers: Maximum number of parallel workers
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        print(f"🚀 Starting bulk data fetch for {len(symbols)} symbols (period: {period})")
        
        all_data = []
        failed_symbols = []
        successful_count = 0
        
        # Split symbols into batches for bulk download
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        def fetch_batch(batch_symbols):
            """Fetch a batch of symbols using yfinance bulk download"""
            batch_data = []
            batch_failed = []
            
            try:
                print(f"  📊 Fetching batch of {len(batch_symbols)} symbols: {', '.join(batch_symbols[:3])}{'...' if len(batch_symbols) > 3 else ''}")
                
                # Use yfinance bulk download for speed
                data = yf.download(batch_symbols, period=period, interval=interval, 
                                 group_by='ticker', threads=True, progress=False)
                
                if data.empty:
                    print(f"  ❌ No data returned for batch")
                    return [], batch_symbols
                
                # Handle single vs multiple symbols
                if len(batch_symbols) == 1:
                    symbol = batch_symbols[0]
                    if not data.empty:
                        processed_data = self._process_single_symbol_data(data, symbol)
                        if processed_data is not None:
                            batch_data.append(processed_data)
                            print(f"  ✅ {symbol}: {len(processed_data)} records")
                        else:
                            batch_failed.append(symbol)
                            print(f"  ❌ {symbol}: processing failed")
                else:
                    # Process multiple symbols
                    for symbol in batch_symbols:
                        if symbol in data.columns.levels[0]:
                            symbol_data = data[symbol]
                            if not symbol_data.empty and not symbol_data.isna().all().all():
                                processed_data = self._process_single_symbol_data(symbol_data, symbol)
                                if processed_data is not None:
                                    batch_data.append(processed_data)
                                    print(f"  ✅ {symbol}: {len(processed_data)} records")
                                else:
                                    batch_failed.append(symbol)
                                    print(f"  ❌ {symbol}: processing failed")
                            else:
                                batch_failed.append(symbol)
                                print(f"  ❌ {symbol}: no data")
                        else:
                            batch_failed.append(symbol)
                            print(f"  ❌ {symbol}: not in response")
                            
            except Exception as e:
                print(f"  💥 Batch fetch error: {e}")
                batch_failed = batch_symbols
            
            return batch_data, batch_failed
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(fetch_batch, batch): batch for batch in symbol_batches}
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_data, batch_failed = future.result()
                    all_data.extend(batch_data)
                    failed_symbols.extend(batch_failed)
                    successful_count += len(batch_data)
                    
                    elapsed = time.time() - start_time
                    print(f"  ⏱️  Batch complete. Progress: {successful_count}/{len(symbols)} successful ({elapsed:.1f}s)")
                    
                except Exception as e:
                    print(f"  💥 Batch processing error: {e}")
                    failed_symbols.extend(batch)
        
        # Summary
        elapsed = time.time() - start_time
        success_rate = (successful_count / len(symbols)) * 100 if symbols else 0
        
        if failed_symbols:
            print(f"❌ Failed symbols ({len(failed_symbols)}): {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['Date'] = pd.to_datetime(combined_data['Date'])
            print(f"🎉 Data fetch complete: {len(combined_data)} records from {successful_count} symbols")
            print(f"📈 Success rate: {success_rate:.1f}% | Time: {elapsed:.1f}s | Rate: {len(combined_data)/elapsed:.0f} records/sec")
            return combined_data
        else:
            print(f"💥 No data fetched from any symbol")
            return pd.DataFrame()

    def _process_single_symbol_data(self, data, symbol):
        """Process data for a single symbol"""
        try:
            if data.empty:
                return None
                
            # Reset index to get Date as a column
            processed = data.reset_index()
            processed['Ticker'] = symbol
            
            # Handle missing Adj Close column (use Close as fallback)
            if 'Adj Close' not in processed.columns and 'Close' in processed.columns:
                processed['Adj Close'] = processed['Close']
                print(f"  ⚠️  {symbol}: Using Close as Adj Close fallback")
            
            # Select only available columns
            required_cols = ['Date', 'Ticker']
            optional_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            available_cols = required_cols + [col for col in optional_cols if col in processed.columns]
            
            return processed[available_cols]
            
        except Exception as e:
            print(f"  💥 {symbol}: Processing error: {e}")
            return None

    def load_historical_data(self) -> pd.DataFrame:
        """Load historical price data"""
        if self.historical_csv_path:
            self.historical_data = pd.read_csv(self.historical_csv_path)
            self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
            return self.historical_data
        else:
            raise ValueError("Historical data path not provided")
    
    def update_historical_data(self, symbols: List[str], save_path: str = None) -> pd.DataFrame:
        """
        Update existing historical data with fresh data from yfinance
        """
        import time
        start_time = time.time()
        
        print(f"📚 Starting historical data update for {len(symbols)} symbols")
        
        # Load existing data
        existing_data = self.load_historical_data() if self.historical_csv_path else pd.DataFrame()
        
        if not existing_data.empty:
            # Find the most recent date in existing data
            latest_date = existing_data['Date'].max()
            days_behind = (datetime.now() - latest_date).days
            existing_records = len(existing_data)
            unique_tickers = existing_data['Ticker'].nunique()
            
            print(f"📊 Existing data: {existing_records:,} records from {unique_tickers} tickers")
            print(f"📅 Latest data: {latest_date.strftime('%Y-%m-%d')} ({days_behind} days behind)")
            
            if days_behind <= 1:
                print("✅ Data is already up to date")
                return existing_data
        else:
            print("📂 No existing historical data found, fetching fresh data")
        
        # Fetch fresh data for the symbols
        print(f"🔄 Fetching fresh data...")
        fresh_data = self.fetch_fresh_data(symbols, period="1y")
        
        if fresh_data.empty:
            print("❌ No fresh data fetched, returning existing data")
            return existing_data if not existing_data.empty else pd.DataFrame()
        
        print(f"📥 Fresh data: {len(fresh_data):,} records from {fresh_data['Ticker'].nunique()} tickers")
        
        if not existing_data.empty:
            print("🔀 Merging with existing data...")
            # Merge with existing data, removing duplicates
            # Keep fresh data for overlapping dates
            combined = pd.concat([existing_data, fresh_data], ignore_index=True)
            
            before_dedup = len(combined)
            # Remove duplicates, keeping the latest (fresh) data
            combined = combined.sort_values(['Ticker', 'Date'])
            combined = combined.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
            after_dedup = len(combined)
            
            print(f"🧹 Removed {before_dedup - after_dedup:,} duplicate records")
        else:
            combined = fresh_data
        
        # Sort by ticker and date
        combined = combined.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Summary statistics
        final_records = len(combined)
        final_tickers = combined['Ticker'].nunique()
        date_range = f"{combined['Date'].min().strftime('%Y-%m-%d')} to {combined['Date'].max().strftime('%Y-%m-%d')}"
        
        print(f"📈 Final dataset: {final_records:,} records from {final_tickers} tickers ({date_range})")
        
        # Save updated data
        if save_path:
            print(f"💾 Saving to {save_path}...")
            combined.to_csv(save_path, index=False)
            file_size_mb = pd.read_csv(save_path).memory_usage(deep=True).sum() / 1024 / 1024
            print(f"✅ Updated historical data saved ({file_size_mb:.1f} MB)")
        
        elapsed = time.time() - start_time
        print(f"⏱️  Update complete in {elapsed:.1f}s")
        
        self.historical_data = combined
        return combined
    
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
    
    def _setup_yfinance_proxy(self):
        """Setup proxy configuration for yfinance to avoid rate limits"""
        try:
            # Setup SOCKS5 proxy via Tor (same as in fetch_nasdaq.py)
            proxy_config = {
                "http": "socks5h://127.0.0.1:9050", 
                "https": "socks5h://127.0.0.1:9050"
            }
            yf.set_config(proxy=proxy_config)
            
            # Create session with curl_cffi for better compatibility
            self.session = curl_cffi.requests.Session(impersonate="chrome")
            print("Proxy configuration applied to yfinance")
            
        except Exception as e:
            print(f"Warning: Failed to setup proxy for yfinance: {e}")
            # Fallback to default session
            self.session = None


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