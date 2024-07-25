import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_candles(candles_data):
    """
    Process raw candle data into a pandas DataFrame.
    
    Args:
    candles_data (dict): Raw candle data from OANDA API
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    if not candles_data or 'candles' not in candles_data:
        logger.error("Invalid candle data received")
        return None

    df = pd.DataFrame([
        {
            "time": candle["time"],
            "open": float(candle["mid"]["o"]),
            "high": float(candle["mid"]["h"]),
            "low": float(candle["mid"]["l"]),
            "close": float(candle["mid"]["c"]),
            "volume": int(candle["volume"])
        }
        for candle in candles_data["candles"]
    ])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    # Add percentage changes
    df['pct_change'] = df['close'].pct_change()
    
    # Add log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df

def prepare_data_for_model(df, sequence_length, train_split=0.8):
    """
    Prepare data for the LSTM model.
    
    Args:
    df (pd.DataFrame): Processed candle data
    sequence_length (int): Number of time steps to look back
    train_split (float): Proportion of data to use for training
    
    Returns:
    tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # Ensure all required columns are present
    features = ['open', 'high', 'low', 'close', 'volume', 'pct_change', 'log_return', 
                'SMA_20', 'EMA_20', 'MACD', 'RSI', 'Stoch_K', 'BBands_Upper', 
                'BBands_Lower', 'ATR', 'OBV']
    
    df = df.dropna()  # Drop rows with NaN values
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[features])

    # Check for NaN or infinite values
    if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        logger.error("NaN or infinite values found in scaled data")
        return None, None, None, None, None

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, features.index('close')])  # Predicting the close price

    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler