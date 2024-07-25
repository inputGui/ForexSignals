import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
        X (numpy.ndarray): Input data of shape (n_samples, sequence_length, n_features)

        Returns:
        numpy.ndarray: Predicted values
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        # Inverse transform the predictions if they were scaled
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

    def predict_next(self, last_sequence):
        """
        Predict the next value given the last sequence of data.

        Args:
        last_sequence (numpy.ndarray): Last sequence of data of shape (1, sequence_length, n_features)

        Returns:
        float: Predicted next value
        """
        prediction = self.predict(last_sequence)
        return prediction[0]


# Usage
if __name__ == "__main__":
    pass
    # Assuming you have a trained model and scaler
    # model = ...
    # scaler = ...

    # predictor = Predictor(model, scaler)

    # Example prediction
    # last_sequence = ... # This should be your last sequence of data
    # next_value = predictor.predict_next(last_sequence)
    # print(f"Predicted next value: {next_value}")