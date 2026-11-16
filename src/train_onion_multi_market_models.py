import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA = Path("data/processed/onion_modeling_dataset_v1.csv")
MODEL_DIR = Path("models/onion_multi_market_v1")
REPORT_DIR = Path("reports")

RESULT_FILE = REPORT_DIR / "onion_multi_market_model_comparison.csv"
PREDICTION_FILE = REPORT_DIR / "onion_multi_market_test_predictions.csv"

if MODEL_DIR.exists():
    raise FileExistsError(
        f"{MODEL_DIR} already exists. Existing models are protected."
    )

if RESULT_FILE.exists() or PREDICTION_FILE.exists():
    raise FileExistsError(
        "An output report already exists. Rename it before running again."
    )

print("Loading modeling dataset...")
df = pd.read_csv(DATA)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

categorical_features = [
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

train = df[df["Dataset_Split"] == "Train"].copy()
test = df[df["Dataset_Split"] == "Test"].copy()

X_train = train[features]
X_test = test[features]
y_train = train[target]
y_test = test[target]

print("Training records:", len(train))
print("Testing records:", len(test))
print("Encoding categorical features...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
            ),
            categorical_features
        ),
        ("numbers", "passthrough", numeric_features)
    ],
    remainder="drop"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)

models = {
    "Linear Regression": LinearRegression(n_jobs=-1),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

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

results = []
prediction_frames = []

print("\n========== MODEL TRAINING ==========")

baseline_predictions = test["Lag_1"].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

results.append({
    "Model": "Previous Price Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2": baseline_r2,
    "Training_Seconds": 0,
    "Status": "Success"
})

prediction_frames.append(pd.DataFrame({
    "State": test["State"].values,
    "District": test["District"].values,
    "Market": test["Market"].values,
    "Variety": test["Variety"].values,
    "Grade": test["Grade"].values,
    "Arrival_Date": test["Arrival_Date"].values,
    "Actual_Price": y_test.values,
    "Predicted_Price": baseline_predictions,
    "Model": "Previous Price Baseline"
}))

print(
    f"Previous Price Baseline | MAE: {baseline_mae:.2f} | "
    f"RMSE: {baseline_rmse:.2f} | R2: {baseline_r2:.4f}"
)

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    try:
        model.fit(X_train_encoded, y_train)

        predictions = model.predict(X_test_encoded)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        elapsed = round(time.time() - start, 2)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Training_Seconds": elapsed,
            "Status": "Success"
        })

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "preprocessor": preprocessor,
                "model": model,
                "categorical_features": categorical_features,
                "numeric_features": numeric_features,
                "target": target
            },
            MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
        )

        prediction_frames.append(pd.DataFrame({
            "State": test["State"].values,
            "District": test["District"].values,
            "Market": test["Market"].values,
            "Variety": test["Variety"].values,
            "Grade": test["Grade"].values,
            "Arrival_Date": test["Arrival_Date"].values,
            "Actual_Price": y_test.values,
            "Predicted_Price": predictions,
            "Model": name
        }))

        print(
            f"{name} | MAE: {mae:.2f} | "
            f"RMSE: {rmse:.2f} | R2: {r2:.4f} | "
            f"Time: {elapsed}s"
        )

    except Exception as error:
        results.append({
            "Model": name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Training_Seconds": round(time.time() - start, 2),
            "Status": f"Failed: {str(error)[:250]}"
        })
        print("MODEL FAILED:", name, error)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    "MAE", na_position="last"
)

comparison.to_csv(RESULT_FILE, index=False)

if prediction_frames:
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )
    predictions_df.to_csv(PREDICTION_FILE, index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

successful = comparison[comparison["Status"] == "Success"]

if len(successful):
    best = successful.sort_values("MAE").iloc[0]
    print("\nLowest MAE model:", best["Model"])
    print("Best MAE:", round(best["MAE"], 2))
    print("Baseline MAE:", round(baseline_mae, 2))

    if best["Model"] != "Previous Price Baseline":
        if best["MAE"] < baseline_mae:
            print("Result: Best ML model beats the baseline.")
        else:
            print("Result: Best ML model does not beat the baseline.")
    else:
        print("Result: Previous-price baseline is the best model.")

print("\nComparison report:", RESULT_FILE)
print("Predictions report:", PREDICTION_FILE)
print("Model folder:", MODEL_DIR)
print("\nMODEL TRAINING COMPLETED")
