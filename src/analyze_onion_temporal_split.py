import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/onion_dataset_deduplicated_v2.csv")
OUTPUT = Path("reports/onion_temporal_split_candidates.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing report safe."
    )

df = pd.read_csv(INPUT)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
df = df.dropna(subset=["Arrival_Date", "Modal_Price"])

group_columns = ["State", "District", "Market", "Variety", "Grade"]

train_start = pd.Timestamp("2025-01-01")
train_end = pd.Timestamp("2025-09-30")
test_start = pd.Timestamp("2025-10-01")
test_end = pd.Timestamp("2025-11-05")

df = df[
    (df["Arrival_Date"] >= train_start) &
    (df["Arrival_Date"] <= test_end)
].copy()

df["Period"] = "Outside"
df.loc[
    df["Arrival_Date"].between(train_start, train_end), "Period"
] = "Train"
df.loc[
    df["Arrival_Date"].between(test_start, test_end), "Period"
] = "Test"

summary = (
    df.groupby(group_columns + ["Period"])
    .agg(Records=("Modal_Price", "count"))
    .reset_index()
)

wide = summary.pivot_table(
    index=group_columns,
    columns="Period",
    values="Records",
    fill_value=0
).reset_index()

for col in ["Train", "Test"]:
    if col not in wide.columns:
        wide[col] = 0

wide["Total_Train_Test"] = wide["Train"] + wide["Test"]

eligible = wide[
    (wide["Total_Train_Test"] >= 100) &
    (wide["Train"] >= 60) &
    (wide["Test"] >= 10)
].copy()

eligible = eligible.sort_values(
    ["Test", "Train"],
    ascending=False
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
eligible.to_csv(OUTPUT, index=False)

print("\n========== CHRONOLOGICAL SPLIT ANALYSIS ==========")
print("Training period:", train_start.date(), "to", train_end.date())
print("Testing period:", test_start.date(), "to", test_end.date())
print("All series in report:", len(wide))
print("Series with at least 60 training observations:", (wide["Train"] >= 60).sum())
print("Series with at least 10 testing observations:", (wide["Test"] >= 10).sum())
print("Eligible series meeting all thresholds:", len(eligible))

print("\nTop 20 eligible series:")
if len(eligible):
    print(
        eligible[
            group_columns + ["Train", "Test", "Total_Train_Test"]
        ].head(20).to_string(index=False)
    )
else:
    print("No series met all thresholds.")

print("\nReport saved:", OUTPUT)
print("TEMPORAL SPLIT ANALYSIS COMPLETED")
