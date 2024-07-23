import pandas as pd
import ta

def add_features(df):
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    
    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    
    # Volatility Indicators
    df['BBands_Upper'], df['BBands_Middle'], df['BBands_Lower'] = ta.volatility.bollinger_hband(df['close']), ta.volatility.bollinger_mavg(df['close']), ta.volatility.bollinger_lband(df['close'])
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    return df