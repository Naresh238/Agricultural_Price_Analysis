import pandas as pd

file = "data/processed/onion_prices.csv"

df = pd.read_csv(file)

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

print("\nONION DATASET INSPECTION")
print("========================")

print("\n1. Dataset shape:")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))

print("\n2. Records by year:")
print(df["Arrival_Date"].dt.year.value_counts().sort_index())

print("\n3. Price statistics:")
print(df[["Min_Price", "Max_Price", "Modal_Price"]].describe().round(2))

print("\n4. Duplicate records:")
print("Duplicates:", df.duplicated().sum())

print("\n5. Unique categorical values:")
for col in ["State", "District", "Market", "Variety", "Grade"]:
    print(f"{col}: {df[col].nunique():,}")

print("\n6. Modal price range:")
print("Minimum:", df["Modal_Price"].min())
print("Maximum:", df["Modal_Price"].max())
print("Average:", round(df["Modal_Price"].mean(), 2))

print("\n7. Records by month:")
print(df["Arrival_Date"].dt.to_period("M").value_counts().sort_index().to_string())

print("\nINSPECTION COMPLETED")
