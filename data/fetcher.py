import requests
import logging
from config import API_URL, HEADERS, OANDA_ACCOUNT_ID, INSTRUMENT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_candles(granularity, count):
    """
    Fetch candlestick data from OANDA API.

    Args:
    granularity (str): The candlestick granularity
    count (int): The number of candles to fetch

    Returns:
    dict: JSON response from the API
    """
    endpoint = f"{API_URL}/v3/instruments/{INSTRUMENT}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"  # Midpoint pricing
    }

    logger.debug(f"Fetching candles for {INSTRUMENT} with granularity {granularity} and count {count}")
    logger.debug(f"API endpoint: {endpoint}")
    logger.debug(f"Request params: {params}")

    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        logger.info(f"Successfully fetched {len(data['candles'])} candles for {INSTRUMENT}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching candles: {e}")
        logger.error(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return None

def get_account_summary():
    """
    Fetch account summary from OANDA API.
    
    Returns:
    dict: JSON response from the API
    """
    endpoint = f"{API_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    
    try:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        logger.info("Successfully fetched account summary")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching account summary: {e}")
        return None
    
def fetch_multiple_instruments(instruments, granularity, count):
    data = {}
    for instrument in instruments:
        data[instrument] = fetch_candles(instrument, granularity, count)
    return data