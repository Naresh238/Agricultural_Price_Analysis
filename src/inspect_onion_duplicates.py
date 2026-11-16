import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_v2.csv")
OUTPUT = Path("reports/onion_duplicate_review.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing report safe."
    )

df = pd.read_csv(INPUT)

key_columns = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade", "Arrival_Date"
]

duplicates = df[
    df.duplicated(subset=key_columns, keep=False)
].copy()

duplicates = duplicates.sort_values(key_columns)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
duplicates.to_csv(OUTPUT, index=False)

print("\n========== DUPLICATE REVIEW ==========")
print("Duplicate rows:", len(duplicates))
print("Duplicate groups:", duplicates.groupby(key_columns).ngroups)
print("Report saved:", OUTPUT)

print("\nFirst 20 duplicate rows:")
print(duplicates.head(20).to_string(index=False))

print("\n========== REVIEW COMPLETED ==========")
