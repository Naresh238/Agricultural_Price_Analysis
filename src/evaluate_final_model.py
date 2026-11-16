import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Loading dataset and trained XGBoost model...")

df = pd.read_csv("data/processed/onion_advanced_features.csv")
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

test = df[df["Arrival_Date"].dt.year == 2026].copy()
test = test.dropna(subset=["Lag_1", "Rolling_Mean_7"]).copy()

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear", "Quarter",
    "Lag_1", "Lag_2", "Rolling_Mean_7",
    "Rolling_Mean_30", "Previous_Price_Change"
]

features = categorical + numeric

X_test = test[features]
y_test = test["Daily_Modal_Price"].to_numpy()

model = joblib.load("models/xgboost_historical_experiment.pkl")

predictions = np.maximum(model.predict(X_test), 0)
previous_predictions = test["Lag_1"].to_numpy()

# Metrics
results = []

for name, predicted in [
    ("XGBoost", predictions),
    ("Previous Price Baseline", previous_predictions)
]:
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R2": r2_score(y_test, predicted)
    })

results_df = pd.DataFrame(results)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    "reports/final_evaluation_metrics.csv",
    index=False
)

# Save individual predictions
prediction_df = pd.DataFrame({
    "Actual_Price": y_test,
    "XGBoost_Prediction": predictions,
    "Previous_Price_Prediction": previous_predictions,
    "XGBoost_Error": y_test - predictions
})

prediction_df.to_csv(
    "reports/xgboost_test_predictions.csv",
    index=False
)

# 1. Actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.25)
plt.xlabel("Actual Onion Price")
plt.ylabel("Predicted Onion Price")
plt.title("Actual vs Predicted Onion Prices - XGBoost")
plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png", dpi=300)
plt.close()

# 2. Prediction error distribution
errors = y_test - predictions

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("XGBoost Prediction Error Distribution")
plt.tight_layout()
plt.savefig("reports/figures/error_distribution.png", dpi=300)
plt.close()

# 3. Residuals vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(predictions, errors, alpha=0.25)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Onion Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Analysis - XGBoost")
plt.tight_layout()
plt.savefig("reports/figures/residual_analysis.png", dpi=300)
plt.close()

# 4. Model metric comparison
comparison = results_df.set_index("Model")[["MAE", "RMSE"]]

comparison.plot(kind="bar", figsize=(9, 6))
plt.ylabel("Error (Rupees)")
plt.title("XGBoost vs Previous Price Baseline")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/model_error_comparison.png", dpi=300)
plt.close()

print("\nFINAL EVALUATION")
print("================")
print(results_df.round(3).to_string(index=False))

print("\nGenerated files:")
print("reports/final_evaluation_metrics.csv")
print("reports/xgboost_test_predictions.csv")
print("reports/figures/actual_vs_predicted.png")
print("reports/figures/error_distribution.png")
print("reports/figures/residual_analysis.png")
print("reports/figures/model_error_comparison.png")

print("\nMODEL EVALUATION COMPLETED")
