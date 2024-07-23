from data.fetcher import fetch_candles
from data.preprocessor import process_candles
from utils.feature_engineering import add_features
from models.train import train_model
from utils.evaluation import evaluate_model

def main():
    # Fetch data
    candles_data = fetch_candles("M15", 5000)
    
    # Process data
    df = process_candles(candles_data)
    df = add_features(df)
    
    # Prepare data for model
    # Split into train/val/test sets
    # Create DataLoaders
    
    # Train model
    model = train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim)
    
    # Evaluate model
    rmse = evaluate_model(model, test_loader)
    print(f"Model RMSE: {rmse}")
    
    # Generate predictions
    # Implement prediction logic here

if __name__ == "__main__":
    main()