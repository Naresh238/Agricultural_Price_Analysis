import pandas as pd
from pathlib import Path

path = Path("reports/onion_multi_market_test_predictions.csv")

df = pd.read_csv(path)

print("FILE:", path)
print("\nSHAPE:", df.shape)
print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nMODEL NAMES:")
for name in sorted(df["Model"].dropna().unique()):
    print("-", name)

print("\nFIRST 5 ROWS:")
print(df.head().to_string(index=False))

print("\nDATE RANGE:")
if "Arrival_Date" in df.columns:
    print(df["Arrival_Date"].min(), "to", df["Arrival_Date"].max())
