import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("reports/onion_multi_market_test_predictions.csv")
SUMMARY_OUTPUT = Path("reports/onion_model_diagnostic_v1.csv")
MARKET_OUTPUT = Path("reports/onion_market_model_comparison_v1.csv")

for path in [SUMMARY_OUTPUT, MARKET_OUTPUT]:
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Existing report is protected."
        )

df = pd.read_csv(INPUT)

keys = ["State", "District", "Market", "Variety", "Grade"]

df["Error"] = df["Predicted_Price"] - df["Actual_Price"]
df["Absolute_Error"] = df["Error"].abs()
df["Squared_Error"] = df["Error"] ** 2

print("\n========== OVERALL MODEL DIAGNOSTICS ==========")

summary = (
    df.groupby("Model")
    .agg(
        Records=("Actual_Price", "count"),
        MAE=("Absolute_Error", "mean"),
        RMSE=("Squared_Error", lambda x: np.sqrt(x.mean())),
        Median_Absolute_Error=("Absolute_Error", "median"),
        P90_Absolute_Error=("Absolute_Error", lambda x: x.quantile(0.90)),
        Mean_Bias=("Error", "mean"),
        Markets=("Market", "nunique")
    )
    .reset_index()
)

summary = summary.sort_values("MAE")
summary.to_csv(SUMMARY_OUTPUT, index=False)

print(summary.round(2).to_string(index=False))

print("\n========== MARKET-LEVEL COMPARISON ==========")

market_scores = (
    df.groupby(keys + ["Model"])
    .agg(
        MAE=("Absolute_Error", "mean"),
        Records=("Actual_Price", "count")
    )
    .reset_index()
)

market_wide = market_scores.pivot_table(
    index=keys,
    columns="Model",
    values="MAE"
).reset_index()

baseline_name = "Previous Price Baseline"

if baseline_name not in market_wide.columns:
    raise ValueError("Previous Price Baseline predictions are missing.")

ml_models = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]

for model in ml_models:
    if model not in market_wide.columns:
        continue

    market_wide[model + "_Improvement"] = (
        market_wide[baseline_name] - market_wide[model]
    )

    wins = (
        market_wide[model + "_Improvement"] > 0
    ).sum()

    losses = (
        market_wide[model + "_Improvement"] < 0
    ).sum()

    print(f"\n{model}")
    print("Markets where ML beats baseline:", wins)
    print("Markets where baseline wins:", losses)
    print(
        "Average market MAE improvement:",
        round(market_wide[model + "_Improvement"].mean(), 2)
    )

market_wide.to_csv(MARKET_OUTPUT, index=False)

print("\n========== BEST MARKETS FOR ML ==========")

if "Gradient Boosting_Improvement" in market_wide.columns:
    print(
        market_wide.sort_values(
            "Gradient Boosting_Improvement",
            ascending=False
        )[
            keys + [
                baseline_name,
                "Gradient Boosting",
                "Gradient Boosting_Improvement"
            ]
        ].head(10).round(2).to_string(index=False)
    )

print("\nSummary report:", SUMMARY_OUTPUT)
print("Market report:", MARKET_OUTPUT)
print("\nMODEL DIAGNOSTICS COMPLETED")
