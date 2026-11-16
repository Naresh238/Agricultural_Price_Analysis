import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DATA = Path("data/processed/onion_model_features.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading feature-engineered dataset...")
df = pd.read_csv(DATA)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Chronological split: train on 2025, test on 2026
train = df[df["Arrival_Date"].dt.year == 2025].copy()
test = df[df["Arrival_Date"].dt.year == 2026].copy()

print("Training records:", f"{len(train):,}")
print("Testing records:", f"{len(test):,}")

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear",
    "Quarter", "Month_Sin", "Month_Cos"
]

features = categorical + numeric
target = "Modal_Price"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Fit category handling only on training data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=100
                ))
            ]),
            categorical
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric
        )
    ]
)

models = {
    "Ridge Regression": Ridge(alpha=10),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=20,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    model_name = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        MODEL_DIR / f"{model_name}.pkl"
    )

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))

# Compare models by RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("================")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]["Model"]
best_file = best_model.lower().replace(" ", "_") + ".pkl"

print("\nBest model by RMSE:", best_model)

joblib.dump(
    joblib.load(MODEL_DIR / best_file),
    MODEL_DIR / "best_onion_price_model.pkl"
)

print("\nMODEL TRAINING COMPLETED")
print("Comparison report: reports/model_comparison.csv")
print("Best model: models/best_onion_price_model.pkl")
