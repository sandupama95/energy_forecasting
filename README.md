# ⚡ Electricity Load Forecasting Pipeline

This repository implements a full machine learning pipeline for time series electricity load forecasting using models such as ARIMA, Prophet, XGBoost, LSTM, and N-BEATS. It uses **Prefect 2.0** for orchestration and **MLflow** for experiment tracking and artifact logging.

---

## 📁 Project Structure

---


## 🚀 Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/electricity-forecasting-prefect.git
cd electricity-forecasting-prefect

```
✅ 2. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
✅ 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Ensure you have PyTorch installed:
```bash
pip install torch
```
✅ 4. Configure MLflow Tracking Server
Start the MLflow UI on a free port (e.g., 5001):
```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```
Visit http://localhost:5001 to monitor experiments.
Set tracking URI in the terminal before running:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
```
Or update it directly in prefect_flow.py:
```bash
mlflow.set_tracking_uri("http://localhost:5001")
```
🧪 Run the Pipeline
🔁 1. Execute the Prefect Flow
```bash
python prefect_flow.py
```
This will:

Load the dataset

Perform feature engineering

Run anomaly detection and clean anomalies

Train selected models

Log results and models to MLflow

Configure the steps and models in configs/config.yaml.

📊 MLflow UI
Access all experiment metrics, plots, models, and artifacts in:

👉 http://localhost:5001

Forecast accuracy (MAE, RMSE, MAPE)

Anomaly detection summaries

Model artifacts: .pkl, .h5, .joblib, .pth


👤 Author
Sandupama Balasuriya
Principal Data Scientist | Electricity Load Forecasting Project
