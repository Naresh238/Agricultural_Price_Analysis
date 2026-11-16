import pandas as pd
from pathlib import Path

source = Path("data/processed/onion_prices.csv")
output = Path("data/processed/onion_prices_cleaned.csv")
audit_file = Path("reports/onion_cleaning_audit.csv")

audit_file.parent.mkdir(parents=True, exist_ok=True)

print("Loading Onion dataset...")
df = pd.read_csv(source)

original_count = len(df)

# Convert date column
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Count exact duplicates before removing them
duplicate_count = int(df.duplicated().sum())

# Remove exact duplicate records
df = df.drop_duplicates().copy()

after_duplicates = len(df)

# Identify records where modal price is outside the
# reported minimum and maximum price range
invalid_range = (
    (df["Modal_Price"] < df["Min_Price"])
    | (df["Modal_Price"] > df["Max_Price"])
)

invalid_range_count = int(invalid_range.sum())

# Remove only records with invalid price relationships
cleaned = df.loc[~invalid_range].copy()

# Keep extreme high and low prices for investigation
cleaned = cleaned.sort_values("Arrival_Date")

cleaned.to_csv(output, index=False)

# Create audit report
audit = pd.DataFrame({
    "Cleaning_Step": [
        "Original records",
        "Exact duplicates removed",
        "Records after duplicate removal",
        "Invalid price-range records removed",
        "Final cleaned records"
    ],
    "Record_Count": [
        original_count,
        duplicate_count,
        after_duplicates,
        invalid_range_count,
        len(cleaned)
    ]
})

audit.to_csv(audit_file, index=False)

print("\nCLEANING COMPLETED")
print("==================")
print("Original records:", f"{original_count:,}")
print("Duplicates removed:", f"{duplicate_count:,}")
print("Records after duplicate removal:", f"{after_duplicates:,}")
print("Invalid price-range records removed:", f"{invalid_range_count:,}")
print("Final cleaned records:", f"{len(cleaned):,}")
print("High prices retained:", int((cleaned["Modal_Price"] > 20000).sum()))
print("Low prices retained:", int((cleaned["Modal_Price"] < 100).sum()))

print("\nCleaned dataset:", output)
print("Cleaning audit:", audit_file)
