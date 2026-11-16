import pandas as pd
from pathlib import Path

SOURCE = Path("data/processed/onion_dataset_deduplicated_v2.csv")
CANDIDATES = Path("reports/onion_temporal_split_candidates.csv")
OUTPUT = Path("data/processed/onion_modeling_dataset_v1.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing dataset safe."
    )

group_columns = ["State", "District", "Market", "Variety", "Grade"]

print("Loading onion dataset...")
df = pd.read_csv(SOURCE)

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"], errors="coerce"
)
df["Modal_Price"] = pd.to_numeric(
    df["Modal_Price"], errors="coerce"
)

df = df.dropna(
    subset=group_columns + ["Arrival_Date", "Modal_Price"]
)

print("Loading eligible market series...")
candidates = pd.read_csv(CANDIDATES)
candidate_keys = candidates[group_columns].drop_duplicates()

df = df.merge(
    candidate_keys,
    on=group_columns,
    how="inner"
)

df = df[
    (df["Arrival_Date"] >= "2025-01-01") &
    (df["Arrival_Date"] <= "2025-11-05")
].copy()

df = df.sort_values(
    group_columns + ["Arrival_Date"]
).reset_index(drop=True)

print("Creating lag features...")

grouped = df.groupby(group_columns, sort=False)["Modal_Price"]

df["Lag_1"] = grouped.shift(1)
df["Lag_2"] = grouped.shift(2)

df["Rolling_Mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(7, min_periods=1).mean()
)

df["Rolling_Mean_30"] = grouped.transform(
    lambda s: s.shift(1).rolling(30, min_periods=1).mean()
)

df["Previous_Price_Change"] = df["Lag_1"] - df["Lag_2"]

df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek
df["DayOfYear"] = df["Arrival_Date"].dt.dayofyear

df = df.dropna(
    subset=["Lag_1", "Lag_2", "Rolling_Mean_7", "Rolling_Mean_30"]
).copy()

df["Dataset_Split"] = "Train"
df.loc[
    df["Arrival_Date"] >= "2025-10-01", "Dataset_Split"
] = "Test"

train = df[df["Dataset_Split"] == "Train"]
test = df[df["Dataset_Split"] == "Test"]

print("\n========== MODELING DATASET SUMMARY ==========")
print("Eligible market series:", len(candidate_keys))
print("Total modeling records:", len(df))
print("Training records:", len(train))
print("Testing records:", len(test))
print("Training markets:", train[group_columns].drop_duplicates().shape[0])
print("Testing markets:", test[group_columns].drop_duplicates().shape[0])

print("\nTraining date range:")
print(train["Arrival_Date"].min(), "to", train["Arrival_Date"].max())

print("\nTesting date range:")
print(test["Arrival_Date"].min(), "to", test["Arrival_Date"].max())

print("\nMissing values in model features:")
features = [
    "Lag_1", "Lag_2", "Rolling_Mean_7",
    "Rolling_Mean_30", "Previous_Price_Change",
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear"
]
print(df[features].isna().sum())

print("\nSample records:")
print(
    df[
        group_columns +
        ["Arrival_Date", "Lag_1", "Lag_2",
         "Rolling_Mean_7", "Rolling_Mean_30",
         "Modal_Price", "Dataset_Split"]
    ].head(10).to_string(index=False)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nDataset saved:", OUTPUT)
print("FEATURE ENGINEERING COMPLETED")
