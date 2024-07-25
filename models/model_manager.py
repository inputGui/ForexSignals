import torch
import logging
import os
from datetime import datetime, timedelta
from models.lstm import LSTMModel
from models.train import train_model
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, model_path='best_forex_model.pth',
                 retrain_interval_minutes=1440):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.model_path = model_path
        self.retrain_interval = timedelta(minutes=retrain_interval_minutes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_or_train_model(self, X, y):
        if self._should_train_new_model():
            logger.info("Training new model...")
            model = self._train_new_model(X, y)
        else:
            logger.info("Loading existing model...")
            model = self._load_existing_model()

        return model

    def _should_train_new_model(self):
        if not os.path.exists(self.model_path):
            return True

        model_time = datetime.fromtimestamp(os.path.getmtime(self.model_path))
        return datetime.now() - model_time > self.retrain_interval

    def _train_new_model(self, X, y):
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Train model
        model = train_model(train_loader, val_loader, self.input_dim, self.hidden_dim,
                            self.layer_dim, self.output_dim)

        # Save model
        torch.save(model.state_dict(), self.model_path)
        logger.info(f"New model saved to {self.model_path}")

        return model

    def _load_existing_model(self):
        model = LSTMModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")


# Usage
if __name__ == "__main__":
    # Example usage
    input_dim = 10  # This should match the number of features in your data
    hidden_dim = 64
    layer_dim = 2
    output_dim = 1

    manager = ModelManager(input_dim, hidden_dim, layer_dim, output_dim)

    # Assuming X and y are your prepared data
    # X, y = ... (get this from your DataPipeline)

    # model = manager.load_or_train_model(X, y)
    # Now you have a trained or loaded model ready for predictions