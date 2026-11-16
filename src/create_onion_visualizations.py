from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COMPARISON_FILE = Path("reports/onion_hybrid_test_comparison.csv")
PREDICTIONS_FILE = Path("reports/onion_hybrid_test_predictions.csv")
MARKET_FILE = Path("reports/onion_hybrid_market_improvement_analysis.csv")

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

outputs = [
    OUTPUT_DIR / "onion_model_mae_comparison.png",
    OUTPUT_DIR / "onion_hybrid_market_results.png",
    OUTPUT_DIR / "onion_actual_vs_predicted.png"
]

for output in outputs:
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Existing chart preserved."
        )

for file in [COMPARISON_FILE, PREDICTIONS_FILE, MARKET_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

comparison = pd.read_csv(COMPARISON_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)
market = pd.read_csv(MARKET_FILE)

# --------------------------------------------------
# CHART 1: MODEL MAE COMPARISON
# --------------------------------------------------

comparison = comparison.sort_values("MAE")

plt.figure(figsize=(11, 6))
plt.bar(comparison["Model"], comparison["MAE"])

plt.title("Onion Price Prediction - Model MAE Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (₹ per quintal)")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(outputs[0], dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# CHART 2: HYBRID VS BASELINE MARKET RESULTS
# --------------------------------------------------

result_counts = (
    market["Hybrid_Result"]
    .value_counts()
    .reindex(["Hybrid Wins", "Baseline Wins", "Tie"], fill_value=0)
)

plt.figure(figsize=(9, 6))
plt.bar(result_counts.index, result_counts.values)

plt.title("Hybrid Strategy vs Baseline - Market Results")
plt.xlabel("Market-level result")
plt.ylabel("Number of market series")

for i, value in enumerate(result_counts.values):
    plt.text(i, value + 5, str(value), ha="center")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(outputs[1], dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# CHART 3: ACTUAL VS PREDICTED PRICES
# --------------------------------------------------

hybrid_predictions = predictions[
    predictions["Model"] == "Hybrid Strategy"
].copy()

hybrid_predictions = hybrid_predictions.dropna(
    subset=["Actual_Price", "Predicted_Price"]
)

if hybrid_predictions.empty:
    raise ValueError("No valid hybrid prediction records found.")

# Use a reproducible sample for a readable scatter plot.
sample = hybrid_predictions.sample(
    n=min(3000, len(hybrid_predictions)),
    random_state=42
)

actual = sample["Actual_Price"].to_numpy()
predicted = sample["Predicted_Price"].to_numpy()

lower = min(actual.min(), predicted.min())
upper = max(actual.max(), predicted.max())

plt.figure(figsize=(8, 8))
plt.scatter(actual, predicted, alpha=0.35, s=15)
plt.plot([lower, upper], [lower, upper], linestyle="--")

plt.title("Onion Prices - Actual vs Predicted")
plt.xlabel("Actual Modal Price (₹ per quintal)")
plt.ylabel("Predicted Modal Price (₹ per quintal)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(outputs[2], dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\nALL THREE CHARTS CREATED SUCCESSFULLY")

for output in outputs:
    print(f"{output} | {output.stat().st_size:,} bytes")

print("\nHybrid prediction records:", len(hybrid_predictions))
print("Scatter sample records:", len(sample))
print("\nMarket results:")
print(result_counts.to_string())

print("\nCharts folder:", OUTPUT_DIR)
