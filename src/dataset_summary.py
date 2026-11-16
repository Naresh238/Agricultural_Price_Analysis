
import pandas as pd
import glob
import os

files = glob.glob("data/raw/agricultural_prices_*.csv")

print("\nAGRICULTURAL DATASET SUMMARY")
print("=" * 60)

total_rows = 0

for file in files:
    print(f"\nFILE: {os.path.basename(file)}")

    rows = 0
    min_date = None
    max_date = None
    missing_count = 0
    duplicate_count = 0
    commodities = set()
    states = set()

    for chunk in pd.read_csv(file, chunksize=100000, low_memory=False):
        rows += len(chunk)

        dates = pd.to_datetime(
            chunk["Arrival_Date"], errors="coerce"
        ).dropna()

        if not dates.empty:
            chunk_min = dates.min()
            chunk_max = dates.max()

            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

        missing_count += int(chunk.isna().sum().sum())

        duplicate_count += int(chunk.duplicated().sum())

        commodities.update(chunk["Commodity"].dropna().unique())
        states.update(chunk["State"].dropna().unique())

    total_rows += rows

    print("Total records:", f"{rows:,}")
    print("Date range:", min_date, "to", max_date)
    print("Missing values:", f"{missing_count:,}")
    print("Within-chunk duplicate rows:", f"{duplicate_count:,}")
    print("Unique commodities:", len(commodities))
    print("Unique states:", len(states))

print("\n" + "=" * 60)
print("TOTAL RECORDS IN ALL FILES:", f"{total_rows:,}")
print("DATASET SUMMARY COMPLETED")
