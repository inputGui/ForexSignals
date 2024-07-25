import logging
from data.fetcher import fetch_candles
from data.preprocessor import process_candles
from utils.feature_engineering import add_features
from sklearn.preprocessing import MinMaxScaler
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, instrument, granularity, count, sequence_length):
        self.instrument = instrument
        self.granularity = granularity
        self.count = count
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    def fetch_and_process_data(self):
        logger.info(f"Fetching data for {self.instrument}")
        candles_data = fetch_candles(self.granularity, self.count)

        if candles_data is None:
            logger.error("Failed to fetch candles data")
            return None

        logger.info("Processing candles data")
        df = process_candles(candles_data)

        logger.info("Adding features")
        df = add_features(df)

        return df

    def prepare_data_for_model(self, df):
        logger.info("Preparing data for model")

        # Check if 'time' is in columns or index
        if 'time' in df.columns:
            features = df.columns.drop('time').tolist()
        else:
            features = df.columns.tolist()
            logger.info("'time' column not found in DataFrame columns. Using all columns as features.")

        logger.info(f"Features being used: {features}")

        scaled_data = self.scaler.fit_transform(df[features])

        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, features.index('close')])

        return np.array(X), np.array(y)

    def run_pipeline(self):
        df = self.fetch_and_process_data()
        if df is not None:
            X, y = self.prepare_data_for_model(df)
            return X, y, self.scaler
        return None, None, None


# Usage
if __name__ == "__main__":
    pipeline = DataPipeline("EUR_USD", "M5", 5000, 60)
    X, y, scaler = pipeline.run_pipeline()
    if X is not None and y is not None:
        logger.info(f"Data prepared: X shape: {X.shape}, y shape: {y.shape}")
    else:
        logger.error("Failed to prepare data")