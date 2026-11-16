import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/onion_prices_cleaned.csv"
OUTPUT_FILE = "data/processed/onion_advanced_features.csv"

print("Loading cleaned onion dataset...")

df = pd.read_csv(INPUT_FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Identify each market and product combination
group_columns = [
    "State", "District", "Market", "Variety", "Grade"
]

# Aggregate multiple same-day observations into one daily price
daily = (
    df.groupby(
        group_columns + ["Arrival_Date"],
        as_index=False,
        dropna=False
    )
    .agg(
        Daily_Modal_Price=("Modal_Price", "mean"),
        Daily_Record_Count=("Modal_Price", "size")
    )
)

daily = daily.sort_values(
    group_columns + ["Arrival_Date"]
).reset_index(drop=True)

print("Daily market records:", f"{len(daily):,}")

# Create historical features using previous observations only
grouped = daily.groupby(group_columns, sort=False)[
    "Daily_Modal_Price"
]

daily["Lag_1"] = grouped.shift(1)
daily["Lag_2"] = grouped.shift(2)

daily["Rolling_Mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=7, min_periods=3
    ).mean()
)

daily["Rolling_Mean_30"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=30, min_periods=7
    ).mean()
)

daily["Previous_Price_Change"] = (
    daily["Lag_1"] - daily["Lag_2"]
)

# Calendar features
daily["Year"] = daily["Arrival_Date"].dt.year
daily["Month"] = daily["Arrival_Date"].dt.month
daily["Day"] = daily["Arrival_Date"].dt.day
daily["DayOfWeek"] = daily["Arrival_Date"].dt.dayofweek
daily["DayOfYear"] = daily["Arrival_Date"].dt.dayofyear
daily["Quarter"] = daily["Arrival_Date"].dt.quarter

print("\nFEATURE ENGINEERING SUMMARY")
print("============================")
print("Original cleaned records:", f"{len(df):,}")
print("Daily aggregated records:", f"{len(daily):,}")
print("Date range:", daily["Arrival_Date"].min(), "to", daily["Arrival_Date"].max())
print("Unique markets:", daily["Market"].nunique())

feature_columns = [
    "Lag_1",
    "Lag_2",
    "Rolling_Mean_7",
    "Rolling_Mean_30",
    "Previous_Price_Change"
]

print("\nMissing historical feature values:")
print(daily[feature_columns].isnull().sum())

# Save the new dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)
daily.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nADVANCED FEATURE ENGINEERING COMPLETED")
