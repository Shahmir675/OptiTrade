"""
Advanced LSTM Model for Multi-Stock Price Prediction
Utilizes technical indicators for enhanced prediction accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class MultiStockLSTM(nn.Module):
    """
    Advanced LSTM model for predicting multiple stock prices
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super(MultiStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep output
        last_output = attn_out[:, -1, :]
        
        # Pass through fully connected layers
        output = last_output
        for layer in self.fc_layers:
            output = layer(output)
            
        return output


class LSTMPredictor:
    """
    High-level wrapper for LSTM-based stock prediction
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_stocks = []
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Adj Close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for LSTM training with robust data validation
        """
        print(f"🧹 Starting data preparation...")
        print(f"📊 Input data shape: {df.shape}")
        
        # Get feature columns (excluding non-numeric columns)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'symbol']]
        
        # Validate target column exists
        if target_col not in feature_cols:
            print(f"❌ Target column '{target_col}' not found in features: {feature_cols}")
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Check for data quality
        print(f"🔍 Checking data quality...")
        total_nulls = df[feature_cols].isnull().sum().sum()
        total_infs = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
        print(f"⚠️  Found {total_nulls} null values and {total_infs} infinite values")
        
        X, y = [], []
        stocks_processed = []
        skipped_stocks = []
        
        # Process each stock individually
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date').copy()
            
            if len(ticker_data) < self.sequence_length + self.prediction_horizon:
                skipped_stocks.append(f"{ticker} (insufficient data: {len(ticker_data)})")
                continue
            
            # Data cleaning for this ticker
            ticker_features = ticker_data[feature_cols].copy()
            
            # Handle missing values
            null_count = ticker_features.isnull().sum().sum()
            if null_count > 0:
                print(f"⚠️  {ticker}: Forward-filling {null_count} null values")
                ticker_features = ticker_features.fillna(method='ffill').fillna(method='bfill')
            
            # Handle infinite values
            inf_mask = np.isinf(ticker_features.select_dtypes(include=[np.number]))
            inf_count = inf_mask.sum().sum()
            if inf_count > 0:
                print(f"⚠️  {ticker}: Clipping {inf_count} infinite values")
                ticker_features = ticker_features.replace([np.inf, -np.inf], np.nan)
                ticker_features = ticker_features.fillna(method='ffill').fillna(method='bfill')
            
            # Final validation - check for remaining issues
            if ticker_features.isnull().any().any():
                print(f"❌ {ticker}: Still has null values after cleaning, skipping")
                skipped_stocks.append(f"{ticker} (uncleaned nulls)")
                continue
                
            ticker_values = ticker_features.values
            
            # Check for extreme values that could break scaling
            if np.any(np.abs(ticker_values) > 1e10):
                print(f"❌ {ticker}: Extreme values detected, skipping")
                skipped_stocks.append(f"{ticker} (extreme values)")
                continue
            
            # Scale features for this ticker
            if ticker not in self.feature_scalers:
                self.feature_scalers[ticker] = MinMaxScaler(feature_range=(0, 1))
            
            try:
                ticker_features_scaled = self.feature_scalers[ticker].fit_transform(ticker_values)
                
                # Validate scaling worked
                if np.any(np.isnan(ticker_features_scaled)) or np.any(np.isinf(ticker_features_scaled)):
                    print(f"❌ {ticker}: Scaling produced NaN/inf values, skipping")
                    skipped_stocks.append(f"{ticker} (scaling failed)")
                    continue
                    
            except Exception as e:
                print(f"❌ {ticker}: Scaling error: {e}")
                skipped_stocks.append(f"{ticker} (scaling error)")
                continue
            
            # Create sequences
            target_idx = feature_cols.index(target_col)
            sequences_added = 0
            
            for i in range(len(ticker_features_scaled) - self.sequence_length - self.prediction_horizon + 1):
                sequence = ticker_features_scaled[i:i + self.sequence_length]
                target = ticker_features_scaled[i + self.sequence_length + self.prediction_horizon - 1, target_idx]
                
                # Validate sequence and target
                if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)) or np.isnan(target) or np.isinf(target):
                    continue
                    
                X.append(sequence)
                y.append(target)
                sequences_added += 1
            
            if sequences_added > 0:
                stocks_processed.append(ticker)
                print(f"✅ {ticker}: {sequences_added} sequences added")
            else:
                print(f"❌ {ticker}: No valid sequences created")
                skipped_stocks.append(f"{ticker} (no valid sequences)")
        
        if skipped_stocks:
            print(f"⚠️  Skipped {len(skipped_stocks)} stocks: {skipped_stocks[:5]}{'...' if len(skipped_stocks) > 5 else ''}")
        
        self.trained_stocks = stocks_processed
        
        if len(X) == 0:
            print(f"💥 No valid training data created!")
            return np.array([]), np.array([]), feature_cols
            
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Final validation
        print(f"🎯 Final data shapes: X={X_array.shape}, y={y_array.shape}")
        print(f"📈 Processed {len(stocks_processed)} stocks with {len(X_array)} total sequences")
        print(f"🔢 Data ranges - X: [{np.min(X_array):.4f}, {np.max(X_array):.4f}], y: [{np.min(y_array):.4f}, {np.max(y_array):.4f}]")
        
        return X_array, y_array, feature_cols
    
    def build_model(self, input_size: int):
        """Build the LSTM model"""
        self.model = MultiStockLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.prediction_horizon,
            dropout=self.dropout
        ).to(self.device)
        
        return self.model
    
    def train(self, 
             X: np.ndarray, 
             y: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.2,
             early_stopping_patience: int = 10):
        """
        Train the LSTM model
        """
        if self.model is None:
            self.build_model(X.shape[2])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"🚀 Training LSTM model on {len(X_train)} samples...")
        print(f"📊 Validation split: {len(X_val)} samples")
        
        # Validate input data doesn't contain NaN/inf
        if torch.isnan(X_train).any() or torch.isinf(X_train).any():
            raise ValueError("Training data contains NaN or infinite values")
        if torch.isnan(y_train).any() or torch.isinf(y_train).any():
            raise ValueError("Training targets contain NaN or infinite values")
        
        pbar = tqdm(range(epochs), desc="LSTM Training", unit="epoch")
        for epoch in pbar:
            # Training phase
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Validate batch doesn't contain NaN/inf
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    print(f"⚠️  Skipping batch {i//batch_size} due to NaN/inf in features")
                    continue
                if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                    print(f"⚠️  Skipping batch {i//batch_size} due to NaN/inf in targets")
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"💥 Model outputs contain NaN/inf at epoch {epoch}, batch {i//batch_size}")
                    print(f"🔍 Batch X stats: min={torch.min(batch_X):.6f}, max={torch.max(batch_X):.6f}")
                    print(f"🔍 Outputs stats: min={torch.min(outputs):.6f}, max={torch.max(outputs):.6f}")
                    break
                
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"💥 Loss is NaN/inf at epoch {epoch}, batch {i//batch_size}")
                    print(f"🔍 Outputs: {outputs.squeeze()[:5]}")
                    print(f"🔍 Targets: {batch_y[:5]}")
                    break
                
                loss.backward()
                
                # Check for NaN gradients
                nan_grads = any(torch.isnan(p.grad).any() if p.grad is not None else False 
                               for p in self.model.parameters())
                if nan_grads:
                    print(f"💥 NaN gradients detected at epoch {epoch}, batch {i//batch_size}")
                    break
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            
            # Check if training should stop due to NaN
            if batch_count == 0:
                print(f"💥 No valid batches processed in epoch {epoch}. Stopping training.")
                break
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i + batch_size]
                    batch_y = y_val[i:i + batch_size]
                    
                    outputs = self.model(batch_X)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue
                        
                    loss = criterion(outputs.squeeze(), batch_y)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                        
                    val_loss += loss.item()
                    val_batch_count += 1
            
            if batch_count > 0 and val_batch_count > 0:
                avg_train_loss = train_loss / batch_count
                avg_val_loss = val_loss / val_batch_count
            else:
                print(f"💥 Invalid loss calculation at epoch {epoch}")
                break
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Update tqdm with current stats
            pbar.set_postfix({
                'Train Loss': f'{avg_train_loss:.6f}',
                'Val Loss': f'{avg_val_loss:.6f}',
                'Best Val': f'{best_val_loss:.6f}',
                'Patience': f'{patience_counter}/{early_stopping_patience}'
            })
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/tmp/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                pbar.write(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('/tmp/best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def predict_stock_prices(self, df: pd.DataFrame, stock_symbols: List[str]) -> Dict[str, float]:
        """
        Predict future prices for specific stocks
        """
        predictions = {}
        
        # Early exit if no stocks were trained
        if not self.trained_stocks:
            print("Warning: No stocks were trained. Returning empty predictions.")
            return predictions
        
        for symbol in stock_symbols:
            if symbol not in self.trained_stocks:
                continue
            
            # Get recent data for the stock
            stock_data = df[df['Ticker'] == symbol].sort_values('Date').tail(self.sequence_length)
            
            if len(stock_data) < self.sequence_length:
                print(f"Warning: Insufficient data for {symbol} ({len(stock_data)}/{self.sequence_length} points). Skipping.")
                continue
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'symbol']]
            features = stock_data[feature_cols].values
            
            # Scale features
            if symbol in self.feature_scalers:
                features_scaled = self.feature_scalers[symbol].transform(features)
            else:
                print(f"Warning: No scaler found for {symbol}. Skipping.")
                continue
            
            # Reshape for prediction
            X = features_scaled.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred_scaled = self.predict(X)[0]
            
            # Inverse transform prediction (assuming first column is price)
            dummy_features = np.zeros((1, len(feature_cols)))
            dummy_features[0, 0] = pred_scaled
            pred_price = self.feature_scalers[symbol].inverse_transform(dummy_features)[0, 0]
            
            predictions[symbol] = float(pred_price)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            },
            'feature_scalers': self.feature_scalers,
            'trained_stocks': self.trained_stocks
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, input_size: int):
        """Load a trained model and scalers"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load configuration
        config = checkpoint['model_config']
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        
        # Build and load model
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scalers
        self.feature_scalers = checkpoint['feature_scalers']
        self.trained_stocks = checkpoint['trained_stocks']
        
        print(f"Model loaded from {filepath}")
        print(f"Trained on {len(self.trained_stocks)} stocks")
        
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(y_test.flatten()))
        pred_direction = np.sign(np.diff(predictions.flatten()))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy)
        }