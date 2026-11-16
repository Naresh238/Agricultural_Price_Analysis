import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_multi_market_coverage.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing report safe."
    )

df = pd.read_csv(INPUT)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
df = df.dropna(subset=["Arrival_Date"])

group_columns = ["State", "District", "Market", "Variety", "Grade"]

coverage = (
    df.groupby(group_columns)
    .agg(
        Total_Records=("Modal_Price", "count"),
        Unique_Dates=("Arrival_Date", "nunique"),
        First_Date=("Arrival_Date", "min"),
        Last_Date=("Arrival_Date", "max"),
        Average_Price=("Modal_Price", "mean"),
    )
    .reset_index()
)

coverage["Average_Price"] = coverage["Average_Price"].round(2)

coverage = coverage.sort_values(
    ["Unique_Dates", "Total_Records"],
    ascending=False
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
coverage.to_csv(OUTPUT, index=False)

print("\n========== MULTI-MARKET COVERAGE ==========")
print("Total market series:", len(coverage))

for threshold in [30, 50, 100, 150, 200]:
    count = (coverage["Unique_Dates"] >= threshold).sum()
    print(f"Series with at least {threshold} observations: {count}")

print("\nTop 20 market series:")
print(
    coverage[
        group_columns +
        ["Unique_Dates", "First_Date", "Last_Date", "Average_Price"]
    ].head(20).to_string(index=False)
)

print("\nReport saved:", OUTPUT)
print("COVERAGE ANALYSIS COMPLETED")
