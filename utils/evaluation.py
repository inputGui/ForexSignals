import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(model.device)
        predictions = model(X_tensor).cpu().numpy()

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse