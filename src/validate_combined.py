import pandas as pd
from pathlib import Path

file = Path("data/processed/agricultural_prices_combined.csv")

total = 0
missing = None
invalid_dates = 0
invalid_prices = 0
min_date = None
max_date = None
states = set()
commodities = set()

for chunk in pd.read_csv(file, chunksize=100000):

    total += len(chunk)

    counts = chunk.isna().sum()
    missing = counts if missing is None else missing.add(counts, fill_value=0)

    dates = pd.to_datetime(chunk["Arrival_Date"], errors="coerce")
    invalid_dates += dates.isna().sum()

    prices = chunk[["Min_Price", "Max_Price", "Modal_Price"]]
    invalid_prices += (
        (prices < 0).any(axis=1) | (~prices.notna().all(axis=1))
    ).sum()

    if dates.notna().any():
        dmin, dmax = dates.min(), dates.max()
        min_date = dmin if min_date is None else min(min_date, dmin)
        max_date = dmax if max_date is None else max(max_date, dmax)

    states.update(chunk["State"].dropna().unique())
    commodities.update(chunk["Commodity"].dropna().unique())

print("\nCOMBINED DATASET VALIDATION")
print("---------------------------")
print("Total records:", f"{total:,}")
print("Earliest date:", min_date)
print("Latest date:", max_date)
print("Invalid dates:", invalid_dates)
print("Invalid or negative prices:", invalid_prices)
print("Number of states:", len(states))
print("Number of commodities:", len(commodities))

print("\nMissing values:")
print(missing.astype(int).to_string())

print("\nDataset validation completed.")
