import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SOURCE = Path("data/processed/onion_modeling_dataset_v1.csv")
SUMMARY = Path("reports/onion_validation_model_comparison.csv")
MARKET_REPORT = Path("reports/onion_validation_market_comparison.csv")

for path in [SUMMARY, MARKET_REPORT]:
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Keeping existing report safe."
        )

print("Loading modeling dataset...")
df = pd.read_csv(SOURCE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

keys = ["State", "District", "Market", "Variety", "Grade"]

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Lag_1", "Lag_2", "Rolling_Mean_7", "Rolling_Mean_30",
    "Previous_Price_Change", "Year", "Month", "Day",
    "DayOfWeek", "DayOfYear"
]

features = categorical + numeric
target = "Modal_Price"

train = df[df["Arrival_Date"] < "2025-09-01"].copy()
valid = df[
    (df["Arrival_Date"] >= "2025-09-01") &
    (df["Arrival_Date"] <= "2025-09-30")
].copy()

print("Training records:", len(train))
print("Validation records:", len(valid))

if train.empty or valid.empty:
    raise ValueError("Training or validation data is empty.")

preprocessor = ColumnTransformer(
    transformers=[
        ("categories",
         OneHotEncoder(handle_unknown="ignore", min_frequency=2),
         categorical),
        ("numbers", "passthrough", numeric)
    ]
)

print("Encoding features...")
X_train = preprocessor.fit_transform(train[features])
X_valid = preprocessor.transform(valid[features])

y_train = train[target]
y_valid = valid[target]

models = {
    "Linear Regression": LinearRegression(n_jobs=-1),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
}

predictions = []

baseline = valid["Lag_1"].to_numpy()

predictions.append(pd.DataFrame({
    **{col: valid[col].values for col in keys},
    "Actual": y_valid.values,
    "Predicted": baseline,
    "Model": "Previous Price Baseline"
}))

print("\nTraining validation models...")

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    predictions.append(pd.DataFrame({
        **{col: valid[col].values for col in keys},
        "Actual": y_valid.values,
        "Predicted": pred,
        "Model": name
    }))

    print("Completed:", name)

all_predictions = pd.concat(predictions, ignore_index=True)
all_predictions["Absolute_Error"] = (
    all_predictions["Actual"] - all_predictions["Predicted"]
).abs()

overall = (
    all_predictions.groupby("Model")
    .agg(
        Records=("Actual", "count"),
        MAE=("Absolute_Error", "mean"),
        Median_Absolute_Error=("Absolute_Error", "median"),
        P90_Absolute_Error=("Absolute_Error", lambda x: x.quantile(0.90))
    )
    .reset_index()
    .sort_values("MAE")
)

market_scores = (
    all_predictions.groupby(keys + ["Model"])
    .agg(
        MAE=("Absolute_Error", "mean"),
        Validation_Records=("Actual", "count")
    )
    .reset_index()
)

market_wide = market_scores.pivot_table(
    index=keys,
    columns="Model",
    values="MAE"
).reset_index()

counts = (
    all_predictions[
        all_predictions["Model"] == "Previous Price Baseline"
    ]
    .groupby(keys)
    .size()
    .reset_index(name="Validation_Records")
)

market_wide = market_wide.merge(counts, on=keys, how="left")

eligible = market_wide[
    market_wide["Validation_Records"] >= 5
].copy()

baseline_name = "Previous Price Baseline"

print("\n========== VALIDATION RESULTS ==========")
print(overall.round(2).to_string(index=False))

print("\nSeries with at least 5 validation observations:", len(eligible))

for name in models:
    if name not in eligible.columns:
        continue

    improvement = eligible[baseline_name] - eligible[name]
    print(f"\n{name}:")
    print("Series where ML beats baseline:", int((improvement > 0).sum()))
    print("Series where baseline wins:", int((improvement < 0).sum()))
    print("Average MAE improvement:", round(improvement.mean(), 2))

overall.to_csv(SUMMARY, index=False)
eligible.to_csv(MARKET_REPORT, index=False)

print("\nSummary report:", SUMMARY)
print("Market validation report:", MARKET_REPORT)
print("\nVALIDATION EXPERIMENT COMPLETED")
