import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file = "data/processed/onion_model_features.csv"
model_file = "models/best_onion_price_model.pkl"

print("Loading dataset and trained model...")
df = pd.read_csv(file)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

train = df[df["Arrival_Date"].dt.year == 2025].copy()
test = df[df["Arrival_Date"].dt.year == 2026].copy()

print("\n1. PRICE DISTRIBUTION BY YEAR")
print("=============================")

for name, data in [("2025 Training", train), ("2026 Testing", test)]:
    prices = data["Modal_Price"]
    print(f"\n{name}")
    print("Records:", f"{len(data):,}")
    print("Mean:", round(prices.mean(), 2))
    print("Median:", round(prices.median(), 2))
    print("95th percentile:", round(prices.quantile(0.95), 2))
    print("99th percentile:", round(prices.quantile(0.99), 2))
    print("Maximum:", round(prices.max(), 2))
    print("Prices above Rs.20,000:", int((prices > 20000).sum()))

categorical = ["State", "District", "Market", "Variety", "Grade"]
numeric = [
    "Month", "Day", "DayOfWeek", "DayOfYear",
    "Quarter", "Month_Sin", "Month_Cos"
]
features = categorical + numeric

X_test = test[features]
y_test = test["Modal_Price"]

model = joblib.load(model_file)
predictions = model.predict(X_test)

print("\n2. BASELINE COMPARISON")
print("======================")

train_median = train["Modal_Price"].median()
train_mean = train["Modal_Price"].mean()

for name, value in [
    ("Training median baseline", train_median),
    ("Training mean baseline", train_mean)
]:
    baseline = np.full(len(y_test), value)
    print(
        name,
        "| MAE:", round(mean_absolute_error(y_test, baseline), 2),
        "| RMSE:", round(np.sqrt(mean_squared_error(y_test, baseline)), 2),
        "| R2:", round(r2_score(y_test, baseline), 4)
    )

print("\n3. RIDGE MODEL PERFORMANCE")
print("===========================")

for name, mask in [
    ("All 2026 test records", np.ones(len(test), dtype=bool)),
    ("Records with price <= Rs.20,000", y_test.to_numpy() <= 20000)
]:
    actual = y_test.to_numpy()[mask]
    predicted = predictions[mask]

    print(f"\n{name}")
    print("Records:", f"{len(actual):,}")
    print("MAE:", round(mean_absolute_error(actual, predicted), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 2))
    print("R2:", round(r2_score(actual, predicted), 4))

print("\n4. MODEL PREDICTION CHECK")
print("=========================")
print("Actual average:", round(y_test.mean(), 2))
print("Predicted average:", round(predictions.mean(), 2))
print("Actual median:", round(y_test.median(), 2))
print("Predicted median:", round(np.median(predictions), 2))
print("Minimum prediction:", round(predictions.min(), 2))
print("Maximum prediction:", round(predictions.max(), 2))

print("\nINVESTIGATION COMPLETED")
