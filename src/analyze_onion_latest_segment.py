import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_latest_segment_analysis.csv")

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

    # Split the series whenever there is a gap greater than 7 days.
    gaps = dates.diff().dt.days.fillna(1)
    segment_ids = gaps.gt(7).cumsum()

    segments = []
    for segment_id, segment_dates in dates.groupby(segment_ids):
        segments.append(segment_dates)

    latest_segment = segments[-1]
    latest_start = latest_segment.min()
    latest_end = latest_segment.max()

    if latest_end.year != 2026:
        continue

    recent_records = group[
        (group["Arrival_Date"] >= latest_start) &
        (group["Arrival_Date"] <= latest_end)
    ]

    results.append({
        "State": values[0],
        "District": values[1],
        "Market": values[2],
        "Variety": values[3],
        "Grade": values[4],
        "Total_Records": len(group),
        "Records_2026": (group["Arrival_Date"].dt.year == 2026).sum(),
        "Latest_Segment_Records": len(latest_segment),
        "Latest_Segment_Start": latest_start.date(),
        "Latest_Segment_End": latest_end.date(),
        "Average_Modal_Price": round(recent_records["Modal_Price"].mean(), 2)
    })

report = pd.DataFrame(results)

if not report.empty:
    report = report.sort_values(
        ["Latest_Segment_Records", "Records_2026"],
        ascending=False
    )

    report.to_csv(OUTPUT, index=False)

    print("\nTOP 20 RECENT CONTINUOUS MARKETS:")
    print(report.head(20).to_string(index=False))
else:
    print("No eligible recent market series found.")

print("\nEligible recent series:", len(report))
print("Report saved:", OUTPUT)
print("\nANALYSIS COMPLETED")
