import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/agricultural_prices_combined_v2.csv")
OUTPUT = Path("data/processed/agricultural_prices_cleaned_v2.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename it before running again."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

total_before = 0
total_after = 0
invalid_price_rows = 0

print("Starting agricultural data cleaning...")

for chunk in pd.read_csv(INPUT, chunksize=200000):

    total_before += len(chunk)

    # Convert price columns to numeric values.
    price_columns = ["Min_Price", "Max_Price", "Modal_Price"]

    for col in price_columns:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Keep only records with valid positive modal prices.
    chunk = chunk.dropna(subset=["Modal_Price"])

    before = len(chunk)

    chunk = chunk[
        (chunk["Modal_Price"] > 0) &
        (chunk["Min_Price"] >= 0) &
        (chunk["Max_Price"] >= 0)
    ].copy()

    # A zero minimum or maximum is treated as unavailable.
    # Replace it with the modal price for consistency.
    chunk.loc[chunk["Min_Price"] == 0, "Min_Price"] = (
        chunk.loc[chunk["Min_Price"] == 0, "Modal_Price"]
    )

    chunk.loc[chunk["Max_Price"] == 0, "Max_Price"] = (
        chunk.loc[chunk["Max_Price"] == 0, "Modal_Price"]
    )

    # Ensure the modal price is inside the available price range.
    # Preserve the original modal price as the prediction target.
    chunk["Min_Price"] = chunk[["Min_Price", "Modal_Price"]].min(axis=1)
    chunk["Max_Price"] = chunk[["Max_Price", "Modal_Price"]].max(axis=1)

    invalid_price_rows += before - len(chunk)

    chunk.to_csv(
        OUTPUT,
        mode="a",
        index=False,
        header=not OUTPUT.exists()
    )

    total_after += len(chunk)

    print(f"Cleaned records: {total_after:,}", end="\r")

print("\n\n========== CLEANING COMPLETED ==========")
print("Original records:", f"{total_before:,}")
print("Cleaned records:", f"{total_after:,}")
print("Rows removed:", f"{total_before - total_after:,}")
print("Output file:", OUTPUT)
print(
    "Output size:",
    round(OUTPUT.stat().st_size / (1024 * 1024), 2),
    "MB"
)
