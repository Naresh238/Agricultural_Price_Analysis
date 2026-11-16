from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "reports" / "onion_multi_market_test_predictions.csv"
OUTPUT_DIR = ROOT / "reports" / "monitoring"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = {
    "Arrival_Date",
    "Actual_Price",
    "Predicted_Price",
    "Model",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
df["Actual_Price"] = pd.to_numeric(df["Actual_Price"], errors="coerce")
df["Predicted_Price"] = pd.to_numeric(df["Predicted_Price"], errors="coerce")

df = df.dropna(subset=list(required))

if df.empty:
    raise ValueError("No valid prediction records available.")

if not np.isfinite(df["Actual_Price"]).all():
    raise ValueError("Actual prices contain infinite values.")

if not np.isfinite(df["Predicted_Price"]).all():
    raise ValueError("Predictions contain infinite values.")

df["Residual"] = df["Actual_Price"] - df["Predicted_Price"]
df["Absolute_Error"] = df["Residual"].abs()

# Overall monitoring metrics for each model.
summary_rows = []

for model_name, group in df.groupby("Model"):
    actual = group["Actual_Price"]
    predicted = group["Predicted_Price"]

    summary_rows.append({
        "Model": model_name,
        "Records": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Mean_Residual": group["Residual"].mean(),
        "Maximum_Absolute_Error": group["Absolute_Error"].max(),
    })

summary = pd.DataFrame(summary_rows).sort_values("MAE")

# Daily monitoring metrics.
daily = (
    df.groupby(["Arrival_Date", "Model"])
    .agg(
        Records=("Actual_Price", "size"),
        MAE=("Absolute_Error", "mean"),
        RMSE=("Residual", lambda x: np.sqrt(np.mean(np.square(x)))),
        Mean_Residual=("Residual", "mean"),
    )
    .reset_index()
    .sort_values(["Arrival_Date", "Model"])
)

# Save the current monitoring snapshot.
run_time = datetime.now().astimezone().isoformat(timespec="seconds")

summary.insert(0, "Monitoring_Run", run_time)
daily.insert(0, "Monitoring_Run", run_time)

summary_file = OUTPUT_DIR / "model_monitoring_summary.csv"
daily_file = OUTPUT_DIR / "daily_model_performance.csv"
history_file = OUTPUT_DIR / "model_monitoring_history.csv"

summary.to_csv(summary_file, index=False)
daily.to_csv(daily_file, index=False)

# Append one overall performance record per model to history.
if history_file.exists():
    history = pd.read_csv(history_file)
    history = pd.concat([history, summary], ignore_index=True)
else:
    history = summary.copy()

history.to_csv(history_file, index=False)

print("\nMODEL MONITORING COMPLETED")
print("Monitoring timestamp:", run_time)
print("Valid prediction records:", len(df))
print("Models monitored:", df["Model"].nunique())

print("\nOVERALL MODEL PERFORMANCE")
print(summary.to_string(index=False))

print("\nMONITORING FILES SAVED")
print(summary_file)
print(daily_file)
print(history_file)
