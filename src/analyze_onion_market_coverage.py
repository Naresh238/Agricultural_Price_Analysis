import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_market_coverage.csv")

df = pd.read_csv(INPUT)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

keys = ["State", "District", "Market", "Variety", "Grade"]

coverage = (
    df.groupby(keys, dropna=False)
    .agg(
        Records=("Modal_Price", "size"),
        Unique_Dates=("Arrival_Date", "nunique"),
        First_Date=("Arrival_Date", "min"),
        Last_Date=("Arrival_Date", "max"),
        Average_Price=("Modal_Price", "mean")
    )
    .reset_index()
)

coverage["Date_Span_Days"] = (
    coverage["Last_Date"] - coverage["First_Date"]
).dt.days

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
coverage.to_csv(OUTPUT, index=False)

print("\n========== MARKET COVERAGE REPORT ==========")
print("Total market-variety-grade series:", len(coverage))

print("\nSeries with at least 30 observations:", (coverage["Records"] >= 30).sum())
print("Series with at least 100 observations:", (coverage["Records"] >= 100).sum())
print("Series with at least 200 observations:", (coverage["Records"] >= 200).sum())
print("Series with at least 300 observations:", (coverage["Records"] >= 300).sum())

print("\nTOP 20 MARKETS BY OBSERVATION COUNT:")
print(
    coverage.sort_values("Records", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nReport saved:", OUTPUT)
print("\nMARKET COVERAGE ANALYSIS COMPLETED")
