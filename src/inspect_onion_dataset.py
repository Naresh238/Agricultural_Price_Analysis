import pandas as pd
from pathlib import Path

file = Path("data/processed/onion_dataset_v2.csv")

print("File exists:", file.exists())

if file.exists():
    print("File size:", round(file.stat().st_size / (1024 * 1024), 2), "MB")

    df = pd.read_csv(file)

    print("\nTotal records:", f"{len(df):,}")
    print("Columns:", list(df.columns))
    print("Date range:", df["Arrival_Date"].min(), "to", df["Arrival_Date"].max())
    print("Unique states:", df["State"].nunique())
    print("Unique commodities:", df["Commodity"].nunique())
    print("\nTop 10 states:")
    print(df["State"].value_counts().head(10))
