# src/prophet_model.py

import pandas as pd
from prophet import Prophet
from src.evaluate import evaluate_forecast

def train_and_predict_prophet(df: pd.DataFrame, forecast_horizon: int = 48, return_model: bool = False):
    """
    1. Prepares df with columns 'ds' (Date) and 'y' (Load).
    2. Splits off last 'forecast_horizon' rows for validation.
    3. Fits Prophet on training.
    4. Creates future DataFrame and forecasts.
    5. Returns (y_pred, y_true, metrics).
    """
    try:
        df_prop = df[["Date", "Load"]].rename(columns={"Date": "ds", "Load": "y"})
        train = df_prop[:-forecast_horizon]
        test = df_prop[-forecast_horizon:]
        holidays_df = None
        if 'PublicHolidays' in df.columns:
            # Filter for rows where 'public holidays' is True (or 1)

            # Create the holidays_df with 'ds' and 'holiday' columns
            # We assign a generic name 'Public Holiday'
            holidays_df = df[df['PublicHolidays'] == 1][['Date']].copy()
            holidays_df.rename(columns={'Date': 'ds'}, inplace=True)
            holidays_df['holiday'] = 'PublicHoliday'

        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, holidays=holidays_df)
        model.fit(train)

        future = model.make_future_dataframe(periods=forecast_horizon, freq="H")
        forecast = model.predict(future)

        y_pred = forecast["yhat"].tail(forecast_horizon).values
        y_true = test["y"].values

        metrics = evaluate_forecast(y_true, y_pred)
        # Save predictions
        forecast_df = pd.DataFrame({
            "timestamp": df["Date"].iloc[-forecast_horizon:],
            "true_load": y_true,
            "predicted_load": y_pred
        })
        forecast_df.to_csv("data/predictions/prophet_predictions.csv", index=False)
        if return_model:
            # Return the trained Prophet object as well
            return y_pred, y_true, metrics, model
        else:
            return y_pred, y_true, metrics
    except Exception as e:
        raise RuntimeError(f"Prophet model training failed: {e}")