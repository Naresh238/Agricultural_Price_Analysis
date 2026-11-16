import pandas as pd

FILE = "data/processed/onion_dataset_deduplicated_v2.csv"

df = pd.read_csv(FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

market = df[
    (df["State"] == "Tamil Nadu") &
    (df["District"] == "Dindigul") &
    (df["Market"] == "Vedasanthur (Uzhavar Sandhai )") &
    (df["Variety"] == "Bellary") &
    (df["Grade"] == "Local")
].copy()

market = market.sort_values("Arrival_Date")

print("\n========== VEDASANTHUR ONION PRICE HISTORY ==========")
print("Total records:", len(market))

print("\nFirst 15 observations:")
print(
    market[["Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]]
    .head(15).to_string(index=False)
)

print("\nLast 20 observations:")
print(
    market[["Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]]
    .tail(20).to_string(index=False)
)

print("\nYear-wise records:")
print(market["Arrival_Date"].dt.year.value_counts().sort_index())

print("\nLatest observation:", market["Arrival_Date"].max().date())
print("\nINSPECTION COMPLETED")
