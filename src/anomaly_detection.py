# src/anomaly_detection.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest

def detect_residual_anomalies(df: pd.DataFrame, resid_thresh: float = 3.0) -> pd.DataFrame:
    """
    1. Seasonal decomposition (period=24) on Load.
    2. Compute rolling std of residual (window=168 hours).
    3. Flag anomalies where |residual| > resid_thresh * rolling_std.
    Returns: df copy with new columns: 'residual', 'anomaly_residual' (bool).
    """
    df = df.copy()
    ts = df.set_index("Date")["Load"].asfreq("H")
    ts = ts.interpolate(method="time")

    decomposition = sm.tsa.seasonal_decompose(ts, model="additive", period=24)
    df["residual"] = decomposition.resid.values

    rolling_std = decomposition.resid.rolling(window=168, min_periods=24).std()
    df["anomaly_residual"] = False
    df.loc[
        df["residual"].abs() > resid_thresh * rolling_std.values,
        "anomaly_residual"
    ] = True

    return df.reset_index(drop=True)

def detect_isolationforest(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """
    1. Select numeric features for anomaly detection:
       Load, Temperature, Cloudiness, Irradiation, lag_1, lag_24, lag_168, load_roll_3h, load_roll_6h, load_roll_24h
    2. Fit IsolationForest(contamination).
    3. Flag anomalies where predicted == -1.
    Returns: df copy with 'anomaly_iforest' (bool).
    """
    df = df.copy()
    feature_cols = []
    for col in ["Load", "Temperature", "Cloudiness", "Irradiation",
                "lag_1", "lag_24", "lag_168", "load_roll_3h", "load_roll_6h", "load_roll_24h"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].dropna()
    iso = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination,
        random_state=42
    )
    iso.fit(X)
    preds = iso.predict(X)  # +1 = normal, -1 = anomaly

    df["anomaly_iforest"] = False
    df.loc[X.index, "anomaly_iforest"] = (preds == -1)
    return df.reset_index(drop=True)

def combine_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine both anomaly flags. 
    New column: 'anomaly_combined' = anomaly_residual OR anomaly_iforest.
    """
    df = df.copy()
    df["anomaly_combined"] = df["anomaly_residual"] | df["anomaly_iforest"]
    return df
