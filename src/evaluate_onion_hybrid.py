import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

VALIDATION_FILE = Path("reports/onion_validation_market_comparison.csv")
TEST_FILE = Path("reports/onion_multi_market_test_predictions.csv")

OUTPUT_COMPARISON = Path("reports/onion_hybrid_test_comparison.csv")
OUTPUT_PREDICTIONS = Path("reports/onion_hybrid_test_predictions.csv")
OUTPUT_MARKETS = Path("reports/onion_hybrid_market_test_comparison.csv")
OUTPUT_SELECTION = Path("reports/onion_hybrid_validation_selection.csv")

outputs = [
    OUTPUT_COMPARISON,
    OUTPUT_PREDICTIONS,
    OUTPUT_MARKETS,
    OUTPUT_SELECTION
]

for path in outputs:
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Existing report kept safe."
        )

keys = ["State", "District", "Market", "Variety", "Grade"]

baseline_name = "Previous Price Baseline"

ml_models = [
    "Linear Regression",
    "Gradient Boosting",
    "XGBoost"
]

print("Loading September validation results...")
validation = pd.read_csv(VALIDATION_FILE)

print("Loading October-November test predictions...")
test = pd.read_csv(TEST_FILE)

required_validation = keys + [
    baseline_name,
    "Validation_Records"
] + ml_models

required_test = keys + [
    "Arrival_Date",
    "Actual_Price",
    "Predicted_Price",
    "Model"
]

for col in required_validation:
    if col not in validation.columns:
        raise ValueError(
            f"Missing validation column: {col}"
        )

for col in required_test:
    if col not in test.columns:
        raise ValueError(
            f"Missing test column: {col}"
        )

test["Arrival_Date"] = pd.to_datetime(
    test["Arrival_Date"],
    errors="coerce"
)

if test["Arrival_Date"].isna().any():
    raise ValueError("Invalid dates found in test predictions.")

print("\nValidation series:", len(validation))
print("Test prediction rows:", len(test))

print("\nChecking test prediction dates...")
print(
    test["Arrival_Date"].min().date(),
    "to",
    test["Arrival_Date"].max().date()
)

# --------------------------------------------------
# STEP 1: Select the best model using September only
# --------------------------------------------------

print("\nSelecting models using validation MAE...")

selection = validation[
    keys + [baseline_name, "Validation_Records"] + ml_models
].copy()

for col in [baseline_name] + ml_models:
    selection[col] = pd.to_numeric(
        selection[col],
        errors="coerce"
    )

selected_models = []
selection_reasons = []

for _, row in selection.iterrows():

    baseline_mae = row[baseline_name]

    available_ml = {
        model: row[model]
        for model in ml_models
        if pd.notna(row[model])
    }

    if pd.isna(baseline_mae) or not available_ml:
        selected_models.append(baseline_name)
        selection_reasons.append("Baseline: insufficient validation")
        continue

    best_ml = min(
        available_ml,
        key=available_ml.get
    )

    if available_ml[best_ml] < baseline_mae:
        selected_models.append(best_ml)
        selection_reasons.append(
            "ML model had lower September validation MAE"
        )
    else:
        selected_models.append(baseline_name)
        selection_reasons.append(
            "Baseline had equal or lower September validation MAE"
        )

selection["Selected_Model"] = selected_models
selection["Selection_Reason"] = selection_reasons

print("\nModel selection counts:")
print(
    selection["Selected_Model"]
    .value_counts()
    .to_string()
)

# --------------------------------------------------
# STEP 2: Apply selected models to unseen test data
# --------------------------------------------------

print("\nApplying selected models to test predictions...")

test_models = set(test["Model"].dropna().unique())

required_test_models = set(ml_models + [baseline_name])

missing_models = required_test_models - test_models

if missing_models:
    raise ValueError(
        f"Missing required test model predictions: {missing_models}"
    )

# Every series must have a baseline prediction.
baseline_test = test[
    test["Model"] == baseline_name
].copy()

if baseline_test.duplicated(keys + ["Arrival_Date"]).any():
    raise ValueError("Duplicate baseline test predictions found.")

# Join the September model selection to October-November ML predictions.
test_ml = test[
    test["Model"].isin(ml_models)
].merge(
    selection[keys + ["Selected_Model"]],
    on=keys,
    how="left",
    validate="many_to_one"
)

# Use ML only when September selected that exact model.
selected_ml = test_ml[
    test_ml["Model"] == test_ml["Selected_Model"]
].copy()

# Use baseline for series where baseline was selected
# or no eligible September validation selection existed.
baseline_keys = selection[
    selection["Selected_Model"] == baseline_name
][keys].drop_duplicates()

baseline_fallback = baseline_test.merge(
    baseline_keys.assign(_use_baseline=True),
    on=keys,
    how="left"
)

baseline_fallback = baseline_fallback[
    baseline_fallback["_use_baseline"].notna()
].drop(columns=["_use_baseline"])

# Also use baseline for any test series not found in validation selection.
known_keys = selection[keys].drop_duplicates()

unmatched_baseline = baseline_test.merge(
    known_keys.assign(_matched=True),
    on=keys,
    how="left"
)

unmatched_baseline = unmatched_baseline[
    unmatched_baseline["_matched"].isna()
].drop(columns=["_matched"])

hybrid = pd.concat(
    [selected_ml, baseline_fallback, unmatched_baseline],
    ignore_index=True
)

hybrid["Model"] = "Hybrid Strategy"
hybrid["Selected_Model"] = np.where(
    hybrid["Model"] == "Hybrid Strategy",
    hybrid.get("Selected_Model", baseline_name),
    hybrid["Model"]
)

# The selected ML rows already have Selected_Model.
# Baseline fallback rows receive the baseline model name.
hybrid["Selected_Model"] = hybrid["Selected_Model"].fillna(
    baseline_name
)

if hybrid.duplicated(keys + ["Arrival_Date"]).any():
    raise ValueError(
        "Hybrid output contains duplicate market-date predictions."
    )

# --------------------------------------------------
# STEP 3: Compare all models and hybrid strategy
# --------------------------------------------------

print("\nCalculating final test metrics...")

test_for_metrics = test.copy()
test_for_metrics["Evaluation_Model"] = test_for_metrics["Model"]

hybrid_for_metrics = hybrid.copy()
hybrid_for_metrics["Evaluation_Model"] = "Hybrid Strategy"

combined = pd.concat(
    [test_for_metrics, hybrid_for_metrics],
    ignore_index=True
)

combined["Absolute_Error"] = (
    combined["Actual_Price"] -
    combined["Predicted_Price"]
).abs()

results = []

for model_name, group in combined.groupby("Evaluation_Model"):

    actual = group["Actual_Price"]
    predicted = group["Predicted_Price"]

    results.append({
        "Model": model_name,
        "Records": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Median_Absolute_Error": group["Absolute_Error"].median(),
        "P90_Absolute_Error": group["Absolute_Error"].quantile(0.90),
        "R2": r2_score(actual, predicted)
    })

comparison = pd.DataFrame(results).sort_values("MAE")

# --------------------------------------------------
# STEP 4: Market-level hybrid performance
# --------------------------------------------------

hybrid["Absolute_Error"] = (
    hybrid["Actual_Price"] - hybrid["Predicted_Price"]
).abs()

market_report = (
    hybrid.groupby(keys)
    .agg(
        Test_Records=("Actual_Price", "count"),
        MAE=("Absolute_Error", "mean")
    )
    .reset_index()
)

# Add selected model for each market series.
market_report = market_report.merge(
    selection[keys + ["Selected_Model"]],
    on=keys,
    how="left"
)

market_report["Selected_Model"] = (
    market_report["Selected_Model"].fillna(baseline_name)
)

# --------------------------------------------------
# STEP 5: Save separate reports
# --------------------------------------------------

comparison.to_csv(OUTPUT_COMPARISON, index=False)
hybrid.to_csv(OUTPUT_PREDICTIONS, index=False)
market_report.to_csv(OUTPUT_MARKETS, index=False)
selection.to_csv(OUTPUT_SELECTION, index=False)

print("\n========== FINAL TEST COMPARISON ==========")
print(comparison.round(2).to_string(index=False))

print("\n========== HYBRID MODEL SELECTION ==========")
print(
    selection["Selected_Model"]
    .value_counts()
    .to_string()
)

print("\nHybrid test prediction records:", len(hybrid))
print("Hybrid market series:", market_report.shape[0])

print("\n========== REPORTS CREATED ==========")
for path in outputs:
    print(path)

print("\nHYBRID EVALUATION COMPLETED")
