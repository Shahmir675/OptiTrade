import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def exponential_moving_avg(df: pd.DataFrame, price_col='Adj Close', window_size=20):
    return df[price_col].ewm(span=window_size, adjust=False).mean()

def macd_line(df: pd.DataFrame, price_col='Adj Close', short_window=12, long_window=26):
    short_ema = exponential_moving_avg(df, price_col, short_window)
    long_ema = exponential_moving_avg(df, price_col, long_window)
    return short_ema - long_ema

def macd_signal(df: pd.DataFrame, price_col='Adj Close', signal_window=9, short_window=12, long_window=26):
    macd = macd_line(df, price_col, short_window, long_window)
    return macd.ewm(span=signal_window, adjust=False).mean()

def scale_group(group):
    scaler = MinMaxScaler(feature_range=(0, 1))
    group.iloc[:, 2:] = scaler.fit_transform(group.iloc[:, 2:])
    return group

def convert_numeric(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].astype(float)
    return df
