import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_continuous_market_series.csv")

df = pd.read_csv(INPUT)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
df = df.dropna(subset=["Arrival_Date"])

keys = ["State", "District", "Market", "Variety", "Grade"]
results = []

for values, group in df.groupby(keys):
    group = group.sort_values("Arrival_Date").copy()

    if len(group) < 100:
        continue

    dates = group["Arrival_Date"].drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.fillna(1)

    # A gap greater than 7 days starts a new segment.
    segment_ids = gaps.gt(7).cumsum()
    segment_sizes = dates.groupby(segment_ids).size()

    largest_segment = int(segment_sizes.max())
    recent_2026 = group[group["Arrival_Date"].dt.year == 2026]

    results.append({
        "State": values[0],
        "District": values[1],
        "Market": values[2],
        "Variety": values[3],
        "Grade": values[4],
        "Total_Records": len(group),
        "Records_2026": len(recent_2026),
        "Largest_Continuous_Segment": largest_segment,
        "First_Date": dates.min().date(),
        "Last_Date": dates.max().date(),
        "Maximum_Gap_Days": int(gaps.max())
    })

report = pd.DataFrame(results)

report = report.sort_values(
    ["Largest_Continuous_Segment", "Total_Records"],
    ascending=False
)

report.to_csv(OUTPUT, index=False)

print("\n========== CONTINUOUS MARKET ANALYSIS ==========")
print("Market series examined:", len(report))

print("\nTOP 20 CONTINUOUS SERIES:")
print(report.head(20).to_string(index=False))

print("\nSeries with at least 100 continuous observations:",
      (report["Largest_Continuous_Segment"] >= 100).sum())

print("\nReport saved:", OUTPUT)
print("\nANALYSIS COMPLETED")
