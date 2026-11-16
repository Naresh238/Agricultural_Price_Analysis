import pandas as pd
from pathlib import Path

KEYS = [
    "State", "District", "Market", "Variety", "Grade"
]

DATE_KEY = "Arrival_Date"

HYBRID_FILE = Path("reports/onion_hybrid_test_predictions.csv")
ALL_MODELS_FILE = Path("reports/onion_multi_market_test_predictions.csv")

OUTPUT = Path("reports/onion_hybrid_market_improvement_analysis.csv")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing report safe."
    )

print("Loading hybrid predictions...")
hybrid = pd.read_csv(HYBRID_FILE)

print("Loading original model predictions...")
original = pd.read_csv(ALL_MODELS_FILE)

hybrid[DATE_KEY] = pd.to_datetime(hybrid[DATE_KEY])
original[DATE_KEY] = pd.to_datetime(original[DATE_KEY])

# Extract baseline predictions only.
baseline = original[
    original["Model"] == "Previous Price Baseline"
].copy()

JOIN_KEYS = KEYS + [DATE_KEY]

# Check for duplicate prediction keys.
if hybrid.duplicated(JOIN_KEYS).any():
    raise ValueError("Duplicate hybrid prediction records detected.")

if baseline.duplicated(JOIN_KEYS).any():
    raise ValueError("Duplicate baseline prediction records detected.")

# Keep only the required baseline columns.
baseline = baseline[
    JOIN_KEYS + ["Actual_Price", "Predicted_Price"]
].rename(columns={
    "Actual_Price": "Baseline_Actual",
    "Predicted_Price": "Baseline_Predicted"
})

hybrid_compare = hybrid[
    JOIN_KEYS + ["Actual_Price", "Predicted_Price"]
].rename(columns={
    "Actual_Price": "Hybrid_Actual",
    "Predicted_Price": "Hybrid_Predicted"
})

print("\nMatching hybrid and baseline test predictions...")

comparison = hybrid_compare.merge(
    baseline,
    on=JOIN_KEYS,
    how="outer",
    indicator=True,
    validate="one_to_one"
)

print("\nMATCHING STATUS:")
print(comparison["_merge"].value_counts().to_string())

if not (comparison["_merge"] == "both").all():
    raise ValueError(
        "Hybrid and baseline predictions do not have identical test records."
    )

# Ensure actual prices agree.
if not (
    comparison["Hybrid_Actual"].round(8)
    == comparison["Baseline_Actual"].round(8)
).all():
    raise ValueError("Actual prices do not match.")

comparison["Hybrid_Error"] = (
    comparison["Hybrid_Actual"] -
    comparison["Hybrid_Predicted"]
).abs()

comparison["Baseline_Error"] = (
    comparison["Baseline_Actual"] -
    comparison["Baseline_Predicted"]
).abs()

# Positive means hybrid has lower error.
comparison["MAE_Improvement"] = (
    comparison["Baseline_Error"] -
    comparison["Hybrid_Error"]
)

# Calculate market-series metrics.
market = (
    comparison.groupby(KEYS)
    .agg(
        Test_Records=("Hybrid_Actual", "count"),
        Baseline_MAE=("Baseline_Error", "mean"),
        Hybrid_MAE=("Hybrid_Error", "mean"),
        Average_Improvement=("MAE_Improvement", "mean")
    )
    .reset_index()
)

market["Hybrid_Result"] = "Tie"

market.loc[
    market["Average_Improvement"] > 0,
    "Hybrid_Result"
] = "Hybrid Wins"

market.loc[
    market["Average_Improvement"] < 0,
    "Hybrid_Result"
] = "Baseline Wins"

market = market.sort_values(
    "Average_Improvement",
    ascending=False
)

# Overall comparison.
print("\n========== OVERALL TEST COMPARISON ==========")

print("Test records:", len(comparison))
print("Market series:", len(market))

print(
    "Baseline MAE:",
    round(comparison["Baseline_Error"].mean(), 2)
)

print(
    "Hybrid MAE:",
    round(comparison["Hybrid_Error"].mean(), 2)
)

print(
    "MAE improvement:",
    round(comparison["MAE_Improvement"].mean(), 2)
)

print(
    "Percentage improvement:",
    round(
        100 * comparison["MAE_Improvement"].mean()
        / comparison["Baseline_Error"].mean(),
        2
    ),
    "%"
)

print("\n========== MARKET-LEVEL RESULTS ==========")

print(
    market["Hybrid_Result"]
    .value_counts()
    .to_string()
)

print("\n========== TOP 10 IMPROVED MARKETS ==========")

print(
    market.head(10).round(2).to_string(index=False)
)

print("\n========== TOP 10 WORSE MARKETS ==========")

print(
    market.tail(10).sort_values(
        "Average_Improvement"
    ).round(2).to_string(index=False)
)

market.to_csv(OUTPUT, index=False)

print("\nReport saved:", OUTPUT)
print("\nMARKET IMPROVEMENT ANALYSIS COMPLETED")
