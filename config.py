import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENVIRONMENT = "fxtrade"  # Changed from "practice" to "fxtrade"

API_URL = f"https://api-{OANDA_ENVIRONMENT}.oanda.com"
HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json",
    "Accept-Datetime-Format": "RFC3339"
}

INSTRUMENT = "GBP_USD"

# Model Configuration
INPUT_DIM = 17  # Number of features
HIDDEN_DIM = 64
LAYER_DIM = 2
OUTPUT_DIM = 1
SEQUENCE_LENGTH = 60  # Number of time steps to look back

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.0001  # Reduced learning rate
NUM_EPOCHS = 100  # Increased number of epochs

# Data Configuration
GRANULARITY = "M15"  # 15-minute candles
CANDLE_COUNT = 5000  # Maximum allowed by OANDA

# Paths
MODEL_SAVE_PATH = "models/forex_model.pth"