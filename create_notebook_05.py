import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

md("""# 05 — Model Training and Comparison

## Agricultural Price Analysis — Onion Multi-Market Forecasting

**Objective:** Document and evaluate the existing trained machine-learning models using the saved test predictions.

**Important:** The models have already been trained and saved. This notebook does not retrain models or overwrite existing model files.
""")

md("""## 1. Libraries and project paths

We use pandas and NumPy for data handling, Matplotlib for visualizations, and scikit-learn metrics to independently verify model performance.""")
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("data/processed/onion_modeling_dataset_v1.csv")
RESULT_FILE = Path("reports/onion_multi_market_model_comparison.csv")
PREDICTION_FILE = Path("reports/onion_multi_market_test_predictions.csv")
MODEL_DIR = Path("models/onion_multi_market_v1")
FIGURE_DIR = Path("reports/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Project paths configured successfully.")
print("Model directory exists:", MODEL_DIR.exists())
""")

md("""## 2. Load the prepared modeling dataset

The target variable is `Modal_Price`. The dataset uses a chronological split:
- Training records: dates before 1 October 2025.
- Testing records: dates from 1 October 2025 onward.

We use the existing prepared dataset without regenerating it.""")
code("""df = pd.read_csv(DATA_FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

train = df[df["Dataset_Split"] == "Train"].copy()
test = df[df["Dataset_Split"] == "Test"].copy()

print("Dataset shape:", df.shape)
print("Training records:", len(train))
print("Testing records:", len(test))
print("Training date range:", train["Arrival_Date"].min(), "to", train["Arrival_Date"].max())
print("Testing date range:", test["Arrival_Date"].min(), "to", test["Arrival_Date"].max())
print("Target variable: Modal_Price")
""")

md("""## 3. Features used by the training script

The actual training script uses five categorical features and ten numerical features.

**Categorical features**
- State
- District
- Market
- Variety
- Grade

**Numerical features**
- Lag_1 and Lag_2
- Rolling_Mean_7 and Rolling_Mean_30
- Previous_Price_Change
- Year, Month, Day, DayOfWeek and DayOfYear

The categorical features were encoded using `OneHotEncoder(handle_unknown="ignore", min_frequency=2)`. Numerical features were passed through unchanged.""")
code("""categorical_features = [
    "State", "District", "Market", "Variety", "Grade"
]

numeric_features = [
    "Lag_1",
    "Lag_2",
    "Rolling_Mean_7",
    "Rolling_Mean_30",
    "Previous_Price_Change",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear"
]

features = categorical_features + numeric_features
target = "Modal_Price"

missing_columns = [c for c in features + [target] if c not in df.columns]

print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numeric_features))
print("Total input features:", len(features))
print("Missing required columns:", missing_columns)

assert not missing_columns, "Required modeling columns are missing."
assert len(train) > 0 and len(test) > 0
""")

md("""## 4. Models and training configurations

The following algorithms were trained using the existing training script:

| Algorithm | Main configuration |
|---|---|
| Linear Regression | `n_jobs=-1` |
| Random Forest | 150 trees, max depth 20, minimum leaf size 2 |
| Gradient Boosting | 100 estimators, learning rate 0.05, max depth 2 |
| XGBoost | 300 estimators, max depth 7, learning rate 0.05 |

All tree-based models used random state 42. XGBoost used the squared-error regression objective and histogram tree method.

A previous-price baseline using `Lag_1` was also evaluated. It is an important benchmark for time-series forecasting.""")
code("""model_files = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib"
}

for model_name, model_path in model_files.items():
    if model_path.exists():
        print(f"{model_name}: saved ({model_path.stat().st_size:,} bytes)")
    else:
        print(f"WARNING — missing model file: {model_path}")
""")

md("""## 5. Load the saved model comparison report

The report contains the metrics and training times recorded during the original training run. We read the existing report instead of running training again.""")
code("""comparison = pd.read_csv(RESULT_FILE)

display(comparison.round(4))

print("Models in report:", len(comparison))
print("Successful runs:", (comparison["Status"] == "Success").sum())
""")

md("""## 6. Independently verify test predictions

The saved prediction file contains the actual and predicted prices for each model. We recalculate MAE, RMSE and R² from those predictions.

- **MAE:** Average absolute prediction error.
- **RMSE:** Square root of average squared prediction error; it penalizes large errors more strongly.
- **R²:** Measures how much variation in the target is explained by the predictions.

The previous-price baseline is included in this comparison.""")
code("""predictions = pd.read_csv(PREDICTION_FILE)
predictions["Arrival_Date"] = pd.to_datetime(predictions["Arrival_Date"])

print("Prediction report shape:", predictions.shape)
print("\\nPrediction counts per model:")
display(predictions.groupby("Model").size().rename("Prediction_Count").to_frame())

verified_rows = []

for model_name, group in predictions.groupby("Model"):
    actual = group["Actual_Price"]
    predicted = group["Predicted_Price"]

    verified_rows.append({
        "Model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Prediction_Count": len(group)
    })

verified = pd.DataFrame(verified_rows)
display(verified.round(4))
""")

md("""## 7. Compare reported and recalculated metrics

This check confirms that the saved prediction report reproduces the metrics in the original model comparison report.""")
code("""check = comparison.merge(
    verified,
    on="Model",
    suffixes=("_Reported", "_Recalculated")
)

for metric in ["MAE", "RMSE", "R2"]:
    check[f"{metric}_Difference"] = (
        check[f"{metric}_Reported"] - check[f"{metric}_Recalculated"]
    )

display(check[[
    "Model",
    "MAE_Reported", "MAE_Recalculated",
    "RMSE_Reported", "RMSE_Recalculated",
    "R2_Reported", "R2_Recalculated"
]].round(6))

for metric in ["MAE", "RMSE", "R2"]:
    assert np.allclose(
        check[f"{metric}_Reported"],
        check[f"{metric}_Recalculated"],
        atol=0.001
    ), f"{metric} verification failed."

print("SUCCESS: Saved metrics match independently recalculated metrics.")
""")

md("""## 8. Visualize model performance

The first chart compares MAE and RMSE. The second chart compares R². Lower MAE and RMSE are preferable; higher R² is preferable.""")
code("""plot_data = verified.set_index("Model")

plot_data[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(11, 5),
    title="Onion Multi-Market Model Error Comparison"
)
plt.ylabel("Prediction Error (price units)")
plt.xlabel("Model")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "onion_model_training_error_comparison.png", dpi=150)
plt.show()

plot_data["R2"].plot(
    kind="bar",
    figsize=(10, 4),
    title="Onion Multi-Market Model R² Comparison"
)
plt.ylabel("R²")
plt.xlabel("Model")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "onion_model_training_r2_comparison.png", dpi=150)
plt.show()

print("Model comparison charts saved in:", FIGURE_DIR)
""")

md("""## 9. Interpretation of results

The verified results show:

- **Previous Price Baseline:** Lowest MAE (approximately 90.21).
- **Gradient Boosting:** Lowest RMSE (approximately 214.91) and highest R² (approximately 0.9490).
- **Linear Regression:** RMSE and R² are very close to Gradient Boosting.
- **XGBoost:** Its test RMSE is higher than Linear Regression and Gradient Boosting.
- **Random Forest:** It has the longest recorded training time in this experiment.

The machine-learning models improve RMSE and R² compared with the previous-price baseline, but none improves the baseline MAE. Model selection therefore depends on which error metric matters most.

These results are from the existing chronological test set. They do not establish that the models will perform equally well on future unseen periods.""")
code("""best_mae = verified.loc[verified["MAE"].idxmin()]
best_rmse = verified.loc[verified["RMSE"].idxmin()]
best_r2 = verified.loc[verified["R2"].idxmax()]

print("Lowest MAE:", best_mae["Model"], round(best_mae["MAE"], 4))
print("Lowest RMSE:", best_rmse["Model"], round(best_rmse["RMSE"], 4))
print("Highest R²:", best_r2["Model"], round(best_r2["R2"], 4))

baseline = verified[verified["Model"] == "Previous Price Baseline"].iloc[0]
ml_results = verified[verified["Model"] != "Previous Price Baseline"]

print("\\nML models with lower MAE than baseline:",
      int((ml_results["MAE"] < baseline["MAE"]).sum()), "out of", len(ml_results))
print("ML models with lower RMSE than baseline:",
      int((ml_results["RMSE"] < baseline["RMSE"]).sum()), "out of", len(ml_results))
""")

md("""## 10. Final conclusion

The multi-market onion forecasting experiment successfully trained and saved four machine-learning models and compared them with a previous-price baseline.

The saved prediction results were independently checked against the original evaluation report. Gradient Boosting achieved the strongest RMSE and R², while the previous-price baseline achieved the lowest MAE.

**No model was retrained in this notebook.** Existing model files, training reports and predictions were used to document and verify the experiment.

**Next stage:** Model evaluation, error analysis, reliability testing and monitoring.""")
code("""print("MODEL TRAINING NOTEBOOK COMPLETED")
print("Dataset:", DATA_FILE)
print("Comparison report:", RESULT_FILE)
print("Prediction report:", PREDICTION_FILE)
print("Saved model directory:", MODEL_DIR)
print("Verification: PASSED")
""")

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python"}
}

output = Path("notebooks/05_Model_Training.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)

if output.exists():
    raise FileExistsError(
        f"{output} already exists. Existing notebook was protected."
    )

nbf.write(nb, output)
nbf.validate(nb)

print("NOTEBOOK CREATED SUCCESSFULLY")
print("File:", output)
print("Total cells:", len(nb.cells))
print("Markdown cells:", sum(c.cell_type == "markdown" for c in nb.cells))
print("Code cells:", sum(c.cell_type == "code" for c in nb.cells))
