import pandas as pd
from pathlib import Path

INPUT = Path("reports/onion_market_coverage.csv")

df = pd.read_csv(INPUT, parse_dates=["First_Date", "Last_Date"])

print("\n========== RECENT MARKET COVERAGE ==========")

print("\n1. Series by year of last observation:")
print(df["Last_Date"].dt.year.value_counts().sort_index())

print("\n2. Series by month of last observation:")
print(
    df["Last_Date"].dt.to_period("M")
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n3. Series with at least 100 records and latest observations:")
eligible = df[df["Records"] >= 100].sort_values(
    ["Last_Date", "Unique_Dates"],
    ascending=[False, False]
)

print("Eligible series:", len(eligible))
print(
    eligible.head(20)[
        ["State", "District", "Market", "Variety", "Grade",
         "Records", "Unique_Dates", "First_Date", "Last_Date"]
    ].to_string(index=False)
)

print("\n4. Series with at least 100 records ending in 2026:")
recent = eligible[eligible["Last_Date"].dt.year == 2026]
print("Count:", len(recent))

print("\nRECENT COVERAGE ANALYSIS COMPLETED")
