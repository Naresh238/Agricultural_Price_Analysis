import pandas as pd
from pathlib import Path

file = Path("data/raw/agricultural_prices_2025.csv")

total_rows = 0
missing = None
min_date = None
max_date = None
invalid_dates = 0

for chunk in pd.read_csv(file, chunksize=100000):
    total_rows += len(chunk)

    counts = chunk.isna().sum()
    missing = counts if missing is None else missing.add(counts, fill_value=0)

    dates = pd.to_datetime(chunk["Arrival_Date"], errors="coerce")
    invalid_dates += dates.isna().sum()

    if dates.notna().any():
        chunk_min = dates.min()
        chunk_max = dates.max()

        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

print("2025 DATASET INSPECTION")
print("-----------------------")
print("Total records:", total_rows)
print("Earliest date:", min_date)
print("Latest date:", max_date)
print("Invalid dates:", invalid_dates)
print("\nMissing values:")
print(missing.astype(int).to_string())
print("\nInspection completed.")
