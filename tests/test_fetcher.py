import unittest
from data.fetcher import fetch_candles
from config import GRANULARITY, CANDLE_COUNT

class TestFetcher(unittest.TestCase):
    def test_fetch_candles(self):
        candles_data = fetch_candles(GRANULARITY, CANDLE_COUNT)
        self.assertIsNotNone(candles_data)
        self.assertIn('candles', candles_data)
        self.assertEqual(len(candles_data['candles']), CANDLE_COUNT)

if __name__ == '__main__':
    unittest.main()