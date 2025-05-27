# src/lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from src.evaluate import evaluate_forecast

def create_sequences(series, n_steps):
    """
    Create lagged sequences of length 'n_steps' for LSTM.
    Input: 1D array of shape (n_samples,).
    Returns: (X, y) where:
      X shape = (n_samples-n_steps, n_steps, 1)
      y shape = (n_samples-n_steps,)
    """
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i : i + n_steps])
        y.append(series[i + n_steps])
    X = np.array(X)
    y = np.array(y)
    # reshape X to [samples, timesteps, features=1]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_and_predict_lstm(df: pd.DataFrame, forecast_horizon: int = 48, lookback: int = 24, return_model: bool = False):
    """
    1. Uses only the 'Load' column.
    2. Scales to [0,1] via MinMaxScaler.
    3. Creates lagged sequences (lookback hours).
    4. Splits off last `forecast_horizon` for testing.
    5. Builds a simple LSTM network and trains it.
    6. Predicts on test, rescales back to original.
    7. Returns (y_pred, y_true, metrics).
    """
    df = df.copy().dropna().reset_index(drop=True)
    series = df["Load"].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series).flatten()

    X_all, y_all = create_sequences(series_scaled, lookback)
    test_start = len(X_all) - forecast_horizon

    X_train, y_train = X_all[:test_start], y_all[:test_start]
    X_test, y_test = X_all[test_start:], y_all[test_start:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation="relu", input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[es])

    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate_forecast(y_true, y_pred)
    # Save predictions
    forecast_df = pd.DataFrame({
        "timestamp": df["Date"].iloc[-forecast_horizon:],
        "true_load": y_true,
        "predicted_load": y_pred
    })
    forecast_df.to_csv("data/predictions/lstm_predictions.csv", index=False)
    if return_model:
        return y_pred, y_true, metrics, model
    else:
        return y_pred, y_true, metrics
