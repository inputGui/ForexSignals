import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm import LSTMModel
import os
import time
import logging
from datetime import datetime, timedelta
import torch.optim.lr_scheduler as lr_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim, num_epochs=50, learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.squeeze(-1))
            if torch.isnan(loss).any():
                logger.error(f"NaN loss encountered. Inputs: {batch_x}, Outputs: {outputs}, Targets: {batch_y}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.squeeze(-1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_forex_model.pth')
            logger.info('Model saved.')

    return model


def load_or_train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim,
                        retrain_interval_minutes=1440):
    """
    Load the existing model if it exists and is recent, otherwise train a new one.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        input_dim (int): Number of input features.
        hidden_dim (int): Number of features in hidden state.
        layer_dim (int): Number of LSTM layers.
        output_dim (int): Number of output features.
        retrain_interval_minutes (int): Interval in minutes for model retraining.

    Returns:
        LSTMModel: Loaded or newly trained model.
    """
    model_path = 'best_forex_model.pth'

    if os.path.exists(model_path):
        model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        if datetime.now() - model_time < timedelta(minutes=retrain_interval_minutes):
            logger.info("Loading existing model...")
            model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
            model.load_state_dict(torch.load(model_path))
            return model

    logger.info("Training new model...")
    return train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim)


# Example usage
if __name__ == "__main__":
    # Assuming you have your data loaders and dimensions set up
    input_dim = 10  # example dimension
    hidden_dim = 64
    layer_dim = 2
    output_dim = 1

    # These should be your actual data loaders
    train_loader = ...
    val_loader = ...

    model = load_or_train_model(train_loader, val_loader, input_dim, hidden_dim, layer_dim, output_dim)
