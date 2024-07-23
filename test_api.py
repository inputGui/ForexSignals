from data.fetcher import fetch_candles, get_account_summary
from data.preprocessor import process_candles
from config import GRANULARITY, CANDLE_COUNT, INSTRUMENT

def main():
    print("Fetching account summary...")
    account_summary = get_account_summary()
    if account_summary:
        print("Account Summary:")
        print(f"Balance: {account_summary['account']['balance']}")
        print(f"Open Trade Count: {account_summary['account']['openTradeCount']}")
    else:
        print("Failed to fetch account summary")

    print("\nFetching candles...")
    candles_data = fetch_candles(GRANULARITY, CANDLE_COUNT)
    
    if candles_data:
        print(f"Successfully fetched {len(candles_data['candles'])} candles for {INSTRUMENT}")
        
        print("Processing candles...")
        df = process_candles(candles_data)
        
        if df is not None:
            print(f"Processed DataFrame shape: {df.shape}")
            print("\nFirst few rows of the DataFrame:")
            print(df.head())
        else:
            print("Failed to process candles")
    else:
        print("Failed to fetch candles")

if __name__ == "__main__":
    main()