import pandas as pd
from pathlib import Path

FILES = [
    Path("data/raw/agricultural_prices_2025.csv"),
    Path("data/raw/agricultural_prices_2026.csv")
]

OUTPUT = Path(
    "data/processed/agricultural_prices_combined_v2.csv"
)

COLUMNS = [
    "State", "District", "Market", "Commodity", "Variety",
    "Grade", "Arrival_Date", "Min_Price", "Max_Price",
    "Modal_Price", "Commodity_Code"
]

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename it before running again."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

total_rows = 0
overlap_removed = 0
invalid_dates = 0
missing_keys = 0

print("Starting safe dataset combination...")

for filename in FILES:

    print(f"\nReading {filename.name}")

    for chunk in pd.read_csv(filename, chunksize=100000):

        if set(chunk.columns) != set(COLUMNS):
            raise ValueError(
                f"Unexpected columns in {filename.name}"
            )

        chunk = chunk[COLUMNS].copy()

        chunk["Arrival_Date"] = pd.to_datetime(
            chunk["Arrival_Date"],
            errors="coerce"
        )

        invalid_dates += chunk["Arrival_Date"].isna().sum()

        chunk = chunk.dropna(subset=["Arrival_Date"])

        # Remove the overlapping date from the 2026 file.
        # Keep the December 30 records from the 2025 file.
        if filename.name == "agricultural_prices_2026.csv":

            before = len(chunk)

            chunk = chunk[
                chunk["Arrival_Date"] >
                pd.Timestamp("2025-12-30")
            ]

            overlap_removed += before - len(chunk)

        # Remove records missing essential identification fields.
        required = [
            "State", "District", "Market", "Commodity"
        ]

        before = len(chunk)
        chunk = chunk.dropna(subset=required)
        missing_keys += before - len(chunk)

        chunk.to_csv(
            OUTPUT,
            mode="a",
            index=False,
            header=not OUTPUT.exists()
        )

        total_rows += len(chunk)

        print(
            f"Records written: {total_rows:,}",
            end="\r"
        )

print("\n\n========== COMBINATION COMPLETED ==========")
print("Total records:", f"{total_rows:,}")
print("Overlapping records removed:", f"{overlap_removed:,}")
print("Invalid dates removed:", f"{invalid_dates:,}")
print("Missing essential fields removed:", f"{missing_keys:,}")
print("Output:", OUTPUT)
print(
    "File size:",
    round(OUTPUT.stat().st_size / (1024 * 1024), 2),
    "MB"
)
