# src/evaluate.py

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(y_true, y_pred):
    """
    Returns a dict with MAE, RMSE, MAPE.
    y_true, y_pred: 1D arrays or Series.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}
