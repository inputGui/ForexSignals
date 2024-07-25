import logging
from data.data_pipeline import DataPipeline
from models.model_manager import ModelManager
from models.predictor import Predictor
from utils.evaluation import evaluate_model
from config import INSTRUMENT, GRANULARITY, CANDLE_COUNT, SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, LAYER_DIM, OUTPUT_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting main function")

    # Initialize DataPipeline
    pipeline = DataPipeline(INSTRUMENT, GRANULARITY, CANDLE_COUNT, SEQUENCE_LENGTH)

    # Fetch and process data
    X, y, scaler = pipeline.run_pipeline()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    if X is None or y is None:
        logger.error("Failed to prepare data. Exiting.")
        return

    # Initialize ModelManager
    model_manager = ModelManager(INPUT_DIM, HIDDEN_DIM, LAYER_DIM, OUTPUT_DIM)

    # Load or train model
    model = model_manager.load_or_train_model(X, y)

    # Initialize Predictor
    predictor = Predictor(model, scaler)

    # Evaluate model
    test_size = int(0.2 * len(X))  # Use last 20% of data for testing
    X_test, y_test = X[-test_size:], y[-test_size:]

    rmse = evaluate_model(model, X_test, y_test)
    logger.info(f"Model RMSE: {rmse}")

    # Generate prediction for the next time step
    last_sequence = X[-1].reshape(1, SEQUENCE_LENGTH, -1)
    next_prediction = predictor.predict_next(last_sequence)
    logger.info(f"Prediction for next time step: {next_prediction}")

    # TODO: Implement logic to take action based on the prediction
    # This could involve placing trades, sending alerts, etc.


if __name__ == "__main__":
    main()