
import pandas as pd
import os
import glob

# Find both agricultural datasets
files = glob.glob("data/raw/agricultural_prices_*.csv")

if not files:
    print("ERROR: No agricultural CSV files found!")
    raise SystemExit

print("AGRICULTURAL DATASETS FOUND")

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", os.path.basename(file))
    print("SIZE:", round(os.path.getsize(file) / (1024 * 1024), 2), "MB")

    # Read sample without loading the entire large file
    df = pd.read_csv(file, nrows=10, low_memory=False)

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nFIRST 10 RECORDS:")
    print(df.to_string(index=False))

print("\nDATASET INSPECTION COMPLETED")
