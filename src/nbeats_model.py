# src/nbeats_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.evaluate import evaluate_forecast


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, basis_function):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        theta = self.fc4(x)
        return self.basis_function(theta)


class TrendBasis:
    def __init__(self, degree, forecast_length):
        self.degree = degree
        self.forecast_length = forecast_length
        t = torch.linspace(-1, 1, forecast_length)
        self.basis = torch.stack([t ** i for i in range(degree)], dim=0)

    def __call__(self, theta):
        return torch.matmul(theta, self.basis)


class NBeatsModel(nn.Module):
    def __init__(self, input_size, forecast_length, hidden_size=128, degree=3):
        super().__init__()
        self.block = NBeatsBlock(
            input_size=input_size,
            theta_size=degree,
            hidden_size=hidden_size,
            basis_function=TrendBasis(degree, forecast_length)
        )

    def forward(self, x):
        return self.block(x)


def create_sequences(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)


def train_and_predict_nbeats(df: pd.DataFrame, forecast_horizon: int = 48, lookback: int = 24, return_model=False):
    df = df.copy().dropna().reset_index(drop=True)
    values = df["Load"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values).flatten()

    X_all, y_all = create_sequences(values_scaled, lookback)
    test_start = len(X_all) - forecast_horizon

    X_train, y_train = X_all[:test_start], y_all[:test_start]
    X_test, y_test = X_all[test_start:], y_all[test_start:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = NBeatsModel(input_size=lookback, forecast_length=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(X_train).squeeze()
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze().numpy()

    y_pred = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate_forecast(y_true, y_pred)
    # Save predictions
    forecast_df = pd.DataFrame({
        "timestamp": df["Date"].iloc[-forecast_horizon:],
        "true_load": y_true,
        "predicted_load": y_pred
    })
    forecast_df.to_csv("data/predictions/nbeat_predictions.csv", index=False)

    if return_model:
        return y_pred, y_true, metrics, model
    else:
        return y_pred, y_true, metrics
