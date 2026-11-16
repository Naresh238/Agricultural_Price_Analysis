import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
output = base / "tableau" / "data"
output.mkdir(parents=True, exist_ok=True)

files = [
    "monthly_summary.csv",
    "state_summary.csv",
    "commodity_summary.csv"
]

for filename in files:
    source = base / "data" / "processed" / filename
    if source.exists():
        df = pd.read_csv(source)
        destination = output / filename
        df.to_csv(destination, index=False)
        print(f"Exported: {filename} ({len(df):,} rows)")
    else:
        print(f"Not found: {source}")

print("Tableau data preparation completed.")
