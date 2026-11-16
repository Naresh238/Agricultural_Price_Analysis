import pandas as pd
from pathlib import Path

file = Path("data/raw/agricultural_prices_2026.csv")

print("FILE SIZE:", round(file.stat().st_size / (1024 * 1024), 2), "MB")

total_rows = 0
missing = None
min_date = None
max_date = None
invalid_dates = 0

for chunk in pd.read_csv(file, chunksize=100000):
    total_rows += len(chunk)

    null_counts = chunk.isna().sum()
    missing = null_counts if missing is None else missing.add(null_counts, fill_value=0)

    dates = pd.to_datetime(chunk["Arrival_Date"], errors="coerce")
    invalid_dates += dates.isna().sum()

    chunk_min = dates.min()
    chunk_max = dates.max()

    if pd.notna(chunk_min):
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)

    if pd.notna(chunk_max):
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)

print("\nTOTAL RECORDS:", total_rows)
print("EARLIEST DATE:", min_date)
print("LATEST DATE:", max_date)
print("INVALID DATES:", invalid_dates)

print("\nMISSING VALUES:")
print(missing.astype(int).to_string())

print("\nCOLUMN DATA TYPES:")
sample = pd.read_csv(file, nrows=5)
print(sample.dtypes.to_string())

print("\nINSPECTION COMPLETED")
