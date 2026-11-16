import pandas as pd
from pathlib import Path

FILE = Path("data/processed/onion_dataset_v2.csv")

df = pd.read_csv(FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

print("\n========== ONION DATA QUALITY REPORT ==========")

print("\n1. Missing values:")
print(df.isnull().sum())

print("\n2. Exact duplicate records:")
print(df.duplicated().sum())

key_columns = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade", "Arrival_Date"
]

print("\n3. Duplicate market-date records:")
print(df.duplicated(subset=key_columns).sum())

print("\n4. Invalid dates:")
print(df["Arrival_Date"].isna().sum())

print("\n5. Price validation:")
print("Negative modal prices:", (df["Modal_Price"] < 0).sum())
print("Zero modal prices:", (df["Modal_Price"] == 0).sum())
print("Modal below minimum:", (df["Modal_Price"] < df["Min_Price"]).sum())
print("Modal above maximum:", (df["Modal_Price"] > df["Max_Price"]).sum())

print("\n6. Unique markets:", df["Market"].nunique())
print("Unique districts:", df["District"].nunique())
print("Unique states:", df["State"].nunique())

print("\n7. Top 10 markets:")
print(df["Market"].value_counts().head(10))

print("\n8. Records by year:")
print(df["Arrival_Date"].dt.year.value_counts().sort_index())

print("\n========== QUALITY CHECK COMPLETED ==========")
