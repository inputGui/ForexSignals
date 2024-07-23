import unittest
import os
from config import API_URL, ACCESS_TOKEN, INSTRUMENT

class TestConfig(unittest.TestCase):
    def test_environment_variables(self):
        self.assertIsNotNone(os.getenv("OANDA_API_KEY"), "OANDA_API_KEY is not set in .env file")
    
    def test_config_variables(self):
        self.assertEqual(API_URL, "https://api-fxpractice.oanda.com")
        self.assertIsNotNone(ACCESS_TOKEN)
        self.assertEqual(INSTRUMENT, "GBP_USD")

if __name__ == '__main__':
    unittest.main()