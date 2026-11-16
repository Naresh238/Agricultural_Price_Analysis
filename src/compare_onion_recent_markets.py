import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_recent_market_candidates.csv")

df = pd.read_csv(INPUT)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
df = df.dropna(subset=["Arrival_Date"])

keys = ["State", "District", "Market", "Variety", "Grade"]

results = []

for values, group in df.groupby(keys):
    group = group.sort_values("Arrival_Date").copy()

    if len(group) < 100 or group["Arrival_Date"].max().year != 2026:
        continue

    gaps = group["Arrival_Date"].diff().dt.days.dropna()
    recent = group[group["Arrival_Date"] >= "2026-01-01"]

    results.append({
        "State": values[0],
        "District": values[1],
        "Market": values[2],
        "Variety": values[3],
        "Grade": values[4],
        "Total_Records": len(group),
        "Records_2026": len(recent),
        "First_Date": group["Arrival_Date"].min().date(),
        "Last_Date": group["Arrival_Date"].max().date(),
        "Average_Gap_Days": round(gaps.mean(), 2) if len(gaps) else None,
        "Maximum_Gap_Days": gaps.max() if len(gaps) else None,
        "Average_Modal_Price": round(group["Modal_Price"].mean(), 2)
    })

candidates = pd.DataFrame(results)

candidates = candidates.sort_values(
    ["Records_2026", "Total_Records"],
    ascending=False
)

candidates.to_csv(OUTPUT, index=False)

print("\n========== RECENT MARKET CANDIDATES ==========")
print("Candidate series:", len(candidates))

print("\nTOP 20 CANDIDATES:")
print(candidates.head(20).to_string(index=False))

print("\nSaved report:", OUTPUT)
print("\nANALYSIS COMPLETED")
