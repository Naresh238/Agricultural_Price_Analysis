import pandas as pd
from pathlib import Path

FILE = Path("data/processed/onion_dataset_deduplicated_v2.csv")

df = pd.read_csv(FILE)

print("\n========== ONION DATASET VALIDATION ==========")
print("Total records:", f"{len(df):,}")
print("Exact duplicate rows:", df.duplicated().sum())
print("Unique states:", df["State"].nunique())
print("Unique markets:", df["Market"].nunique())

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

print("Invalid dates:", df["Arrival_Date"].isna().sum())
print("Start date:", df["Arrival_Date"].min().date())
print("End date:", df["Arrival_Date"].max().date())

print("\nRecords by year:")
print(df["Arrival_Date"].dt.year.value_counts().sort_index())

print("\nVALIDATION COMPLETED")
