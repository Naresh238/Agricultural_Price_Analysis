import pandas as pd
from pathlib import Path

source = Path("data/processed/agricultural_prices_combined.csv")
output = Path("data/processed/onion_prices.csv")

parts = []

print("Extracting Onion records...")

for chunk in pd.read_csv(source, chunksize=100000):
    onion = chunk[
        chunk["Commodity"].str.strip().str.lower() == "onion"
    ]

    if not onion.empty:
        parts.append(onion)

df = pd.concat(parts, ignore_index=True)

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
df = df.sort_values("Arrival_Date")

df.to_csv(output, index=False)

print("\nONION DATASET CREATED")
print("----------------------")
print("Total records:", f"{len(df):,}")
print("Start date:", df["Arrival_Date"].min())
print("End date:", df["Arrival_Date"].max())
print("Unique states:", df["State"].nunique())
print("Unique districts:", df["District"].nunique())
print("Unique markets:", df["Market"].nunique())
print("Average modal price:", round(df["Modal_Price"].mean(), 2))
print("Output:", output)
