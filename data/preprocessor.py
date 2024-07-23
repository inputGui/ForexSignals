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
    logger.info(f"Processed {len(df)} candles into DataFrame")
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
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error("Missing required columns in DataFrame")
        return None

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[required_columns])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 3])  # Predicting the close price

    X, y = np.array(X), np.array(y)

    # Split the data into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info(f"Prepared data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train, X_test, y_test, scaler