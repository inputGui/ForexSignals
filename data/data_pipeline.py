from data.fetcher import fetch_candles
from data.preprocessor import process_candles, prepare_data_for_model
from utils.feature_engineering import add_features
from config import INSTRUMENT, GRANULARITY, CANDLE_COUNT, SEQUENCE_LENGTH

def prepare_data():
    # Fetch data
    candles_data = fetch_candles(GRANULARITY, CANDLE_COUNT)
    
    # Process candles
    df = process_candles(candles_data)
    
    # Add features
    df = add_features(df)
    
    # Prepare for model
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_model(df, SEQUENCE_LENGTH)
    
    return X_train, y_train, X_test, y_test, scaler

def prepare_multi_timeframe_data(timeframes):
    data = {}
    for tf in timeframes:
        candles_data = fetch_candles(tf, CANDLE_COUNT)
        df = process_candles(candles_data)
        df = add_features(df)
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_model(df, SEQUENCE_LENGTH)
        data[tf] = (X_train, y_train, X_test, y_test, scaler)
    return data