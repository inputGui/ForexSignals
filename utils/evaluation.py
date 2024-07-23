import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    return rmse