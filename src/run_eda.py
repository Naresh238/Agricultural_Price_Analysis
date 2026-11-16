import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

source = Path("data/processed/agricultural_prices_combined.csv")
output = Path("data/processed")
figures = Path("reports/figures")

output.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

samples = []
commodity_parts = []
state_parts = []
monthly_parts = []

print("Starting EDA on the full dataset...")

for chunk in pd.read_csv(source, chunksize=100000):

    chunk["Arrival_Date"] = pd.to_datetime(chunk["Arrival_Date"])

    # Keep a sample for detailed visual analysis
    samples.append(chunk.sample(
        n=min(1500, len(chunk)),
        random_state=42
    ))

    # Commodity price summaries
    commodity_parts.append(
        chunk.groupby("Commodity")["Modal_Price"]
        .agg(["sum", "count"])
    )

    # State price summaries
    state_parts.append(
        chunk.groupby("State")["Modal_Price"]
        .agg(["sum", "count"])
    )

    # Monthly price summaries
    chunk["Year_Month"] = chunk["Arrival_Date"].dt.to_period("M").astype(str)

    monthly_parts.append(
        chunk.groupby("Year_Month")["Modal_Price"]
        .agg(["sum", "count"])
    )

    print("Processed:", f"{len(samples) * 100000:,}", "rows approximately", end="\r")

# Save sample
sample_df = pd.concat(samples, ignore_index=True)
sample_df.to_csv(output / "eda_sample.csv", index=False)

# Combine summaries from all chunks
def combine_summary(parts):
    combined = pd.concat(parts).groupby(level=0).sum()
    combined["Average_Modal_Price"] = combined["sum"] / combined["count"]
    return combined.sort_values("Average_Modal_Price", ascending=False)

commodity_summary = combine_summary(commodity_parts)
state_summary = combine_summary(state_parts)
monthly_summary = combine_summary(monthly_parts)

commodity_summary.to_csv(output / "commodity_summary.csv")
state_summary.to_csv(output / "state_summary.csv")
monthly_summary.to_csv(output / "monthly_summary.csv")

# Chart 1: Top 10 commodities by average modal price
top_commodities = commodity_summary.head(10).sort_values("Average_Modal_Price")

plt.figure(figsize=(12, 6))
top_commodities["Average_Modal_Price"].plot(kind="barh")
plt.title("Top 10 Commodities by Average Modal Price")
plt.xlabel("Average Modal Price")
plt.tight_layout()
plt.savefig(figures / "top_10_commodities.png")
plt.close()

# Chart 2: Average modal price by state
top_states = state_summary.head(10).sort_values("Average_Modal_Price")

plt.figure(figsize=(12, 6))
top_states["Average_Modal_Price"].plot(kind="barh")
plt.title("Top 10 States by Average Modal Price")
plt.xlabel("Average Modal Price")
plt.tight_layout()
plt.savefig(figures / "top_10_states.png")
plt.close()

# Chart 3: Monthly average modal price
plt.figure(figsize=(12, 6))
monthly_summary["Average_Modal_Price"].sort_index().plot(marker="o")
plt.title("Monthly Average Agricultural Modal Price")
plt.xlabel("Month")
plt.ylabel("Average Modal Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(figures / "monthly_price_trend.png")
plt.close()

print("\n\nEDA COMPLETED SUCCESSFULLY")
print("Sample records:", f"{len(sample_df):,}")

print("\nTOP 10 COMMODITIES:")
print(commodity_summary.head(10)[["Average_Modal_Price", "count"]].to_string())

print("\nTOP 10 STATES:")
print(state_summary.head(10)[["Average_Modal_Price", "count"]].to_string())

print("\nMONTHLY PRICE SUMMARY:")
print(monthly_summary[["Average_Modal_Price", "count"]].to_string())

print("\nSaved files:")
print("- data/processed/eda_sample.csv")
print("- data/processed/commodity_summary.csv")
print("- data/processed/state_summary.csv")
print("- data/processed/monthly_summary.csv")
print("- reports/figures/top_10_commodities.png")
print("- reports/figures/top_10_states.png")
print("- reports/figures/monthly_price_trend.png")
