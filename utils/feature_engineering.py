import pandas as pd
import ta

def add_features(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    # Add more features as needed
    return df