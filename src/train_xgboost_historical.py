import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

print("Loading advanced onion features...")

df = pd.read_csv("data/processed/onion_advanced_features.csv")
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

train = df[df["Arrival_Date"].dt.year == 2025].copy()
test = df[df["Arrival_Date"].dt.year == 2026].copy()

required = ["Lag_1", "Rolling_Mean_7"]
train = train.dropna(subset=required).copy()
test = test.dropna(subset=required).copy()

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear", "Quarter",
    "Lag_1", "Lag_2", "Rolling_Mean_7",
    "Rolling_Mean_30", "Previous_Price_Change"
]

features = categorical + numeric

X_train = train[features]
y_train = train["Daily_Modal_Price"]

X_test = test[features]
y_test = test["Daily_Modal_Price"]

print("Training records:", f"{len(train):,}")
print("Testing records:", f"{len(test):,}")

preprocessor = ColumnTransformer([
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
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric
    )
])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    ))
])

model = TransformedTargetRegressor(
    regressor=xgb_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

print("\nTraining XGBoost with historical features...")
model.fit(X_train, y_train)

predictions = np.maximum(model.predict(X_test), 0)

def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

results = []

results.append(evaluate(
    "Previous Price Baseline",
    y_test,
    test["Lag_1"].to_numpy()
))

results.append(evaluate(
    "Previous 7-Record Average",
    y_test,
    test["Rolling_Mean_7"].to_numpy()
))

results.append(evaluate(
    "XGBoost Historical Features",
    y_test,
    predictions
))

results_df = pd.DataFrame(results).sort_values("RMSE")

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

results_df.to_csv(
    "reports/xgboost_historical_comparison.csv",
    index=False
)

joblib.dump(
    model,
    "models/xgboost_historical_experiment.pkl"
)

print("\nFINAL COMPARISON")
print("================")
print(results_df.to_string(index=False))

print("\nSaved: models/xgboost_historical_experiment.pkl")
print("Saved: reports/xgboost_historical_comparison.csv")
print("\nXGBOOST EXPERIMENT COMPLETED")
