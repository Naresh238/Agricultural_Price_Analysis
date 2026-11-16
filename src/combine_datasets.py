import pandas as pd
from pathlib import Path

file_2025 = Path("data/raw/agricultural_prices_2025.csv")
file_2026 = Path("data/raw/agricultural_prices_2026.csv")
output = Path("data/processed/agricultural_prices_combined.csv")

columns = [
    "State", "District", "Market", "Commodity", "Variety",
    "Grade", "Arrival_Date", "Min_Price", "Max_Price",
    "Modal_Price", "Commodity_Code"
]

output.parent.mkdir(parents=True, exist_ok=True)

if output.exists():
    output.unlink()

total_rows = 0

print("Combining 2025 and 2026 datasets...")

for filename in [file_2025, file_2026]:
    print(f"\nProcessing: {filename.name}")

    for chunk in pd.read_csv(filename, chunksize=100000):
        if list(chunk.columns) != columns:
            raise ValueError(f"Unexpected columns in {filename.name}")

        chunk["Arrival_Date"] = pd.to_datetime(
            chunk["Arrival_Date"], errors="coerce"
        )

        if filename == file_2026:
            chunk = chunk[
                chunk["Arrival_Date"] > pd.Timestamp("2025-12-30")
            ]

        chunk = chunk.dropna(subset=columns)

        chunk.to_csv(
            output,
            mode="a",
            index=False,
            header=not output.exists() or output.stat().st_size == 0
        )

        total_rows += len(chunk)

        print(f"Processed records: {total_rows:,}", end="\r")

print("\n\nCOMBINATION COMPLETED")
print("Total records:", f"{total_rows:,}")
print("Output file:", output)
print("Output size:", round(output.stat().st_size / (1024 * 1024), 2), "MB")
