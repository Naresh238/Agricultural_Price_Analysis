import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/agricultural_prices_cleaned_v2.csv")
OUTPUT = Path("data/processed/onion_dataset_v2.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing file safe."
    )

total_onion = 0

print("Extracting onion records...")

for chunk in pd.read_csv(INPUT, chunksize=200000):

    onion = chunk[
        chunk["Commodity"].astype(str).str.strip().str.casefold() == "onion"
    ].copy()

    if not onion.empty:
        onion.to_csv(
            OUTPUT,
            mode="a",
            index=False,
            header=not OUTPUT.exists()
        )

        total_onion += len(onion)

print("\n========== ONION EXTRACTION COMPLETED ==========")
print("Total onion records:", f"{total_onion:,}")
print("Output file:", OUTPUT)

if OUTPUT.exists():
    print(
        "File size:",
        round(OUTPUT.stat().st_size / (1024 * 1024), 2),
        "MB"
    )
