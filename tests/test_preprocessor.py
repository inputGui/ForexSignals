import unittest
import pandas as pd
from data.preprocessor import process_candles, prepare_data_for_model
from config import SEQUENCE_LENGTH

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a sample candles_data dictionary
        self.sample_candles_data = {
            'candles': [
                {'time': '2021-01-01T00:00:00Z', 'mid': {'o': '1.0', 'h': '1.1', 'l': '0.9', 'c': '1.05'}, 'volume': 100},
                {'time': '2021-01-01T00:15:00Z', 'mid': {'o': '1.05', 'h': '1.15', 'l': '1.0', 'c': '1.1'}, 'volume': 150},
                # Add more sample candles as needed
            ]
        }

    def test_process_candles(self):
        df = process_candles(self.sample_candles_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_candles_data['candles']))
        self.assertListEqual(list(df.columns), ['open', 'high', 'low', 'close', 'volume'])

    def test_prepare_data_for_model(self):
        df = process_candles(self.sample_candles_data)
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_model(df, SEQUENCE_LENGTH)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(scaler)

if __name__ == '__main__':
    unittest.main()