import numpy as np
import pandas as pd
from pathlib import Path

SOURCE = Path(
    "reports/onion_hybrid_market_improvement_analysis.csv"
)

OUTPUT = Path(
    "reports/onion_hybrid_robustness_analysis.csv"
)

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing report safe."
    )

df = pd.read_csv(SOURCE)

required = [
    "Test_Records",
    "Baseline_MAE",
    "Hybrid_MAE",
    "Average_Improvement"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

# Positive improvement means hybrid is better.
improvement = df["Average_Improvement"].to_numpy()
records = df["Test_Records"].to_numpy()

total_records = records.sum()

weighted_improvement = (
    improvement * records
).sum() / total_records

print("\n========== ROBUSTNESS ANALYSIS ==========")

print("Market series:", len(df))
print("Test records:", total_records)

print(
    "Record-weighted improvement:",
    round(weighted_improvement, 4)
)

print(
    "Average market improvement:",
    round(improvement.mean(), 4)
)

print(
    "Median market improvement:",
    round(np.median(improvement), 4)
)

print(
    "Markets with positive improvement:",
    int((improvement > 0).sum())
)

print(
    "Markets with negative improvement:",
    int((improvement < 0).sum())
)

print(
    "Markets with zero improvement:",
    int((improvement == 0).sum())
)

# Cluster bootstrap: resample complete market series.
rng = np.random.default_rng(42)

iterations = 10000
n = len(df)

bootstrap_results = np.empty(iterations)

print("\nRunning", iterations, "market-level bootstrap samples...")

for i in range(iterations):
    selected = rng.integers(0, n, size=n)

    sampled_records = records[selected]
    sampled_improvement = improvement[selected]

    bootstrap_results[i] = (
        (sampled_improvement * sampled_records).sum()
        / sampled_records.sum()
    )

lower = np.percentile(bootstrap_results, 2.5)
upper = np.percentile(bootstrap_results, 97.5)

print("\n========== 95% BOOTSTRAP INTERVAL ==========")
print("Lower bound:", round(lower, 4))
print("Upper bound:", round(upper, 4))

if lower > 0:
    conclusion = (
        "The bootstrap interval is above zero. "
        "This supports a positive hybrid improvement."
    )
elif upper < 0:
    conclusion = (
        "The bootstrap interval is below zero. "
        "This supports a negative hybrid improvement."
    )
else:
    conclusion = (
        "The bootstrap interval includes zero. "
        "The improvement is not clearly distinguishable from zero."
    )

print("\nCONCLUSION:")
print(conclusion)

summary = pd.DataFrame([{
    "Market_Series": n,
    "Test_Records": total_records,
    "Weighted_MAE_Improvement": weighted_improvement,
    "Mean_Market_Improvement": improvement.mean(),
    "Median_Market_Improvement": np.median(improvement),
    "Bootstrap_95_Lower": lower,
    "Bootstrap_95_Upper": upper,
    "Bootstrap_Iterations": iterations,
    "Conclusion": conclusion
}])

summary.to_csv(OUTPUT, index=False)

print("\nReport saved:", OUTPUT)
print("\nROBUSTNESS ANALYSIS COMPLETED")
