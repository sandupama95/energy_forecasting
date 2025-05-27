# prefect_flow.py

import os
import mlflow
from prefect import flow, task, get_run_logger
import pandas as pd

from src.utils import load_config, init_logger
from src.data_loader import load_raw_data
from src.preprocess import engineer_features
from src.anomaly_detection import (
    detect_residual_anomalies,
    detect_isolationforest,
    combine_anomalies
)
from src.evaluate import evaluate_forecast
from src.prophet_model import train_and_predict_prophet
from src.xgboost_model import train_and_predict_xgboost
from src.lstm_model import train_and_predict_lstm

# Set MLflow tracking URI (you can customize or set via env var)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))

@task
def data_quality_checks(df: pd.DataFrame):
    """
    1. Count missing values per column
    2. Log missing counts and % to MLflow
    3. Save a CSV artifact with missing summary
    """
    logger = get_run_logger()
    missing_counts = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    # Log metrics in MLflow
    for col, cnt in missing_counts.items():
        mlflow.log_metric(f"missing_count_{col}", int(cnt))
        mlflow.log_metric(f"missing_pct_{col}", float(missing_pct[col]))

    # Save missing summary as CSV and log as artifact
    missing_df = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values
    })
    missing_path = "missing_summary.csv"
    missing_df.to_csv(missing_path, index=False)
    mlflow.log_artifact(missing_path, artifact_path="data_quality")
    os.remove(missing_path)

    logger.info("Completed data quality checks and logged missingness to MLflow.")
    return df

@task
def run_anomaly_detection(df: pd.DataFrame, resid_thresh: float, contamination: float):
    """
    1. Run residual‐based anomaly detection
    2. Run IsolationForest-based anomaly detection
    3. Combine flags
    4. Log anomaly counts, save timestamps as CSV artifact
    """
    logger = get_run_logger()

    df_resid = detect_residual_anomalies(df, resid_thresh=resid_thresh)
    num_resid = df_resid["anomaly_residual"].sum()
    mlflow.log_metric("num_resid_anomalies", int(num_resid))

    df_if = detect_isolationforest(df_resid, contamination=contamination)
    num_if = df_if["anomaly_iforest"].sum()
    mlflow.log_metric("num_iforest_anomalies", int(num_if))

    df_combined = combine_anomalies(df_if)
    num_comb = df_combined["anomaly_combined"].sum()
    mlflow.log_metric("num_combined_anomalies", int(num_comb))

    # Save anomaly timestamps
    anom_ts = df_combined[df_combined["anomaly_combined"]][["Date", "Load"]]
    anom_path = "anomaly_timestamps.csv"
    anom_ts.to_csv(anom_path, index=False)
    mlflow.log_artifact(anom_path, artifact_path="anomalies")
    os.remove(anom_path)

    logger.info(
        f"Residual anomalies: {num_resid}, IsolationForest: {num_if}, Combined: {num_comb}"
    )
    return df_combined

@task
def clean_anomalies(df: pd.DataFrame, drop: bool = True):
    """
    If drop=True, remove rows where anomaly_combined == True.
    If drop=False, impute Load at anomalies via time-based interpolation.
    """
    logger = get_run_logger()
    if drop:
        df_clean = df[~df["anomaly_combined"]].reset_index(drop=True)
        mlflow.log_metric("rows_after_dropping_anomalies", df_clean.shape[0])
        logger.info(f"Dropped anomalies; new row count = {df_clean.shape[0]}")
        return df_clean
    else:
        df_imputed = df.copy()
        idxs = df_imputed[df_imputed["anomaly_combined"]].index
        df_imputed.loc[idxs, "Load"] = pd.NA
        df_imputed["Load"] = df_imputed["Load"].interpolate(method="time")
        mlflow.log_metric("rows_imputed_anomalies", len(idxs))
        logger.info(f"Imputed {len(idxs)} anomaly rows via interpolation.")
        return df_imputed

@task
def train_and_evaluate(df: pd.DataFrame, forecast_horizon: int, chosen_models: list):
    """
    Trains each model in chosen_models on df, logs metrics/artifacts to MLflow.
    Uses last `forecast_horizon` hours as validation.
    """
    logger = get_run_logger()
    df = df.copy().dropna().reset_index(drop=True)
    # 1. Prophet
    if "prophet" in chosen_models:
        with mlflow.start_run(run_name="Prophet", nested=True):
            y_pred_prophet, y_true, metrics, model_prophet = train_and_predict_prophet(df, forecast_horizon, return_model=True)
            for k, v in metrics.items():
                mlflow.log_metric(f"Prophet_{k}", v)
            # Plot forecast vs actual
            df_prop = df[["Date", "Load"]].rename(columns={"Date": "ds", "Load": "y"})
            df_test = df_prop.tail(forecast_horizon).reset_index(drop=True)
    
            forecast_df = pd.DataFrame({
                "ds": df_test["ds"],
                "actual": df_test["y"],
                "predicted": y_pred_prophet
            })
            fig = forecast_df.plot(x="ds", y=["actual", "predicted"], figsize=(10, 4)).get_figure()
            fig.savefig("prophet_forecast_vs_actual.png")
            mlflow.log_artifact("prophet_forecast_vs_actual.png", artifact_path="plots")
            os.remove("prophet_forecast_vs_actual.png")
            logger.info(f"Prophet metrics: {metrics}")
            mlflow.log_artifact(model_prophet, artifact_path="models/prophet")
            logger.info("Prophet model artifact logged to MLflow.")

    # 2. XGBoost
    if "xgboost" in chosen_models:
        with mlflow.start_run(run_name="XGBoost", nested=True):
            # Modify train_and_predict_xgboost to also return the trained model object
            y_pred_xgboost, y_true, metrics, model_xgboost = train_and_predict_xgboost(df, forecast_horizon, return_model=True)
            for k, v in metrics.items():
                mlflow.log_metric(f"XGBoost_{k}", v)
            logger.info(f"XGBoost metrics: {metrics}")
            # Optionally, log the trained model
            # import joblib
            # joblib.dump(model, "xgboost_model.pkl")
            # mlflow.log_artifact("xgboost_model.pkl", artifact_path="models")
            # Log the XGBoost model to MLflow
            mlflow.xgboost.log_model(model_xgboost, artifact_path="models/xgboost")
            logger.info("XGBoost model artifact logged to MLflow.")

    # 3. LSTM
    if "lstm" in chosen_models:
        with mlflow.start_run(run_name="LSTM", nested=True):
            y_pred_lstm, y_true, metrics, model_lstm = train_and_predict_lstm(df, forecast_horizon, lookback=24, return_model=True)
            for k, v in metrics.items():
                mlflow.log_metric(f"LSTM_{k}", v)
            logger.info(f"LSTM metrics: {metrics}")
            # Optionally, save model weights:
            # model.save("lstm_model.h5")
            # mlflow.log_artifact("lstm_model.h5", artifact_path="models")
            # Log Keras model to MLflow
            mlflow.keras.log_model(model_lstm, artifact_path="models/lstm")
            logger.info("LSTM model artifact logged to MLflow.")

@flow(log_prints=True)
def energy_forecasting_flow(config_path: str = "configs/config.yaml"):
    """
    1. Load config & init logger
    2. Start an MLflow run (parent)
    3. Load raw data
    4. Data quality checks (log to MLflow)
    5. Feature engineering
    6. Anomaly detection (log counts to MLflow)
    7. Clean anomalies (drop/impute, log to MLflow)
    8. Train & evaluate models (log metrics + plots)
    """
    # 1. Load config and init logger
    config = load_config(config_path)
    logger = init_logger(config["log_path"])

    # 2. Parent MLflow run
    with mlflow.start_run(run_name="energy_forecasting_pipeline") as parent_run:
        # Log pipeline parameters
        mlflow.log_params({
            "forecast_horizon": config["forecast_horizon"],
            "resid_thresh": config["anomaly"]["resid_thresh"],
            "contamination": config["anomaly"]["contamination"],
            "drop_anomalies": config["anomaly"]["drop_anomalies"],
            "models": ",".join(config["models"])
        })

        # 3. Load raw data
        logger.info("Loading raw data...")
        df_raw = load_raw_data(config["data_path"])

        # 4. Data quality checks
        df_q = data_quality_checks(df_raw)

        # 5. Feature engineering
        logger.info("Engineering features...")
        df_feat = engineer_features(df_q)

        # 6. Anomaly detection
        logger.info("Running anomaly detection...")
        df_anom = run_anomaly_detection(
            df_feat,
            resid_thresh=config["anomaly"]["resid_thresh"],
            contamination=config["anomaly"]["contamination"]
        )

        # 7. Clean anomalies
        logger.info("Cleaning anomalies before training...")
        df_clean = clean_anomalies(df_anom, drop=config["anomaly"]["drop_anomalies"])

        # 8. Train & evaluate models
        logger.info("Training and evaluating models...")
        train_and_evaluate(
            df_clean,
            forecast_horizon=config["forecast_horizon"],
            chosen_models=config["models"]
        )

        logger.info("Pipeline run complete. Check MLflow UI for metrics & artifacts.")
