import pandas as pd
import numpy as np
from pathlib import Path

source = Path("data/processed/onion_prices_cleaned.csv")
output = Path("data/processed/onion_model_features.csv")

print("Loading cleaned Onion dataset...")
df = pd.read_csv(source)

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Date-based features
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.day_of_week
df["DayOfYear"] = df["Arrival_Date"].dt.day_of_year
df["Quarter"] = df["Arrival_Date"].dt.quarter

# Cyclical month features
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Keep Arrival_Date for chronological splitting.
# Do not use it directly as a model input.
# Do not include Min_Price, Max_Price, or target-derived flags.

df = df.drop(columns=["Min_Price", "Max_Price"])

df = df.sort_values("Arrival_Date")
df.to_csv(output, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("Records:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("Target: Modal_Price")
print("Date retained for time-based splitting")
print("Saved:", output)
