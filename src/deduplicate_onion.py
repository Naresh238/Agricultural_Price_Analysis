import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_v2.csv")
OUTPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping it safe."
    )

print("Loading Onion dataset...")

df = pd.read_csv(INPUT)

before = len(df)

df = df.drop_duplicates(keep="first").copy()

after = len(df)

df.to_csv(OUTPUT, index=False)

print("\n========== DEDUPLICATION REPORT ==========")
print("Original records:", f"{before:,}")
print("Records after deduplication:", f"{after:,}")
print("Duplicate rows removed:", f"{before - after:,}")
print("Output file:", OUTPUT)
print("File size:", round(OUTPUT.stat().st_size / (1024 * 1024), 2), "MB")
print("\nDEDUPLICATION COMPLETED")
