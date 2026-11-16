import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading onion dataset...")

df = pd.read_csv("data/processed/onion_model_features.csv")
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

train = df[df["Arrival_Date"].dt.year == 2025].copy()
test = df[df["Arrival_Date"].dt.year == 2026].copy()

# Sensitivity experiment: exclude extreme training prices only
train = train[train["Modal_Price"] <= 20000].copy()

categorical = ["State", "District", "Market", "Variety", "Grade"]

numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear",
    "Quarter", "Month_Sin", "Month_Cos"
]

features = categorical + numeric

X_train = train[features]
y_train = train["Modal_Price"]

X_test = test[features]
y_test = test["Modal_Price"]

print("\nTRAINING DATA")
print("Records:", f"{len(train):,}")
print("Testing records:", f"{len(test):,}")
print("Training median:", round(y_train.median(), 2))

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
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        )
    ]
)

ridge_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=10))
])

model = TransformedTargetRegressor(
    regressor=ridge_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

print("\nTraining log-transformed Ridge Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Agricultural prices cannot be negative
predictions = np.maximum(predictions, 0)

def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print("Records:", f"{len(actual):,}")
    print("MAE:", round(mean_absolute_error(actual, predicted), 2))
    print(
        "RMSE:",
        round(np.sqrt(mean_squared_error(actual, predicted)), 2)
    )
    print("R2:", round(r2_score(actual, predicted), 4))

# Baselines calculated using the experiment's training data
median_prediction = np.full(len(y_test), y_train.median())
mean_prediction = np.full(len(y_test), y_train.mean())

print("\nBASELINE COMPARISON")
evaluate("Training median baseline", y_test, median_prediction)
evaluate("Training mean baseline", y_test, mean_prediction)

print("\nNEW MODEL PERFORMANCE")
evaluate("Log-transformed Ridge - all 2026 records", y_test, predictions)

normal_mask = y_test.to_numpy() <= 20000

evaluate(
    "Log-transformed Ridge - prices <= Rs.20,000",
    y_test.to_numpy()[normal_mask],
    predictions[normal_mask]
)

print("\nPREDICTION CHECK")
print("Actual average:", round(y_test.mean(), 2))
print("Predicted average:", round(predictions.mean(), 2))
print("Actual median:", round(y_test.median(), 2))
print("Predicted median:", round(np.median(predictions), 2))
print("Minimum prediction:", round(predictions.min(), 2))
print("Maximum prediction:", round(predictions.max(), 2))

Path("reports").mkdir(exist_ok=True)

# Save this as an experimental candidate, not the final model
joblib.dump(
    model,
    "models/log_ridge_experiment.pkl"
)

print("\nEXPERIMENT COMPLETED")
print("Experimental model: models/log_ridge_experiment.pkl")
