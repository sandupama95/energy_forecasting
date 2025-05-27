# src/xgboost_model.py

import pandas as pd
import xgboost as xgb
from src.evaluate import evaluate_forecast

def train_and_predict_xgboost(df: pd.DataFrame, forecast_horizon: int = 48, return_model: bool = False):
    """
    1. Uses all engineered features (except Date and Load).
    2. Splits off last 'forecast_horizon' rows for validation.
    3. Fits XGBRegressor on training.
    4. Predicts on test.
    5. Returns (y_pred, y_true, metrics).
    """
    try:
        df = df.copy().reset_index(drop=True)
        # Drop any remaining NaNs
        df = df.dropna().reset_index(drop=True)

        # feature_cols = [c for c in df.columns if c not in ["Date", "Load"]]
        feature_cols = [
        col for col in df.columns
        if col not in ["Date", "Load"] and df[col].dtype not in ["object"]
        ]
        X = df[feature_cols]
        y = df["Load"]

        X_train = X.iloc[:-forecast_horizon, :]
        y_train = y.iloc[:-forecast_horizon]
        X_test = X.iloc[-forecast_horizon:, :]
        y_test = y.iloc[-forecast_horizon:]

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_true = y_test.values

        metrics = evaluate_forecast(y_true, y_pred)
        # Save predictions
        forecast_df = pd.DataFrame({
            "timestamp": df["Date"].iloc[-forecast_horizon:],
            "true_load": y_true,
            "predicted_load": y_pred
        })
        forecast_df.to_csv("data/predictions/xgboost_predictions.csv", index=False)
        if return_model:
            return y_pred, y_true, metrics, model
        else:
            return y_pred, y_true, metrics
    except Exception as e:
        raise RuntimeError(f"XGBoost model training failed: {e}")