import pandas as pd
from pathlib import Path

REPORT = Path("reports/onion_final_evaluation_report.md")

FILES = {
    "comparison": Path("reports/onion_hybrid_test_comparison.csv"),
    "predictions": Path("reports/onion_hybrid_test_predictions.csv"),
    "selection": Path("reports/onion_hybrid_validation_selection.csv"),
    "robustness": Path("reports/onion_hybrid_robustness_analysis.csv"),
    "market": Path("reports/onion_hybrid_market_improvement_analysis.csv")
}

if REPORT.exists():
    raise FileExistsError(
        f"{REPORT} already exists. Existing report preserved."
    )

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")

comparison = pd.read_csv(FILES["comparison"])
predictions = pd.read_csv(FILES["predictions"])
selection = pd.read_csv(FILES["selection"])
robustness = pd.read_csv(FILES["robustness"])
market = pd.read_csv(FILES["market"])

predictions["Arrival_Date"] = pd.to_datetime(
    predictions["Arrival_Date"]
)

start_date = predictions["Arrival_Date"].min().date()
end_date = predictions["Arrival_Date"].max().date()

hybrid = comparison[
    comparison["Model"] == "Hybrid Strategy"
].iloc[0]

baseline = comparison[
    comparison["Model"] == "Previous Price Baseline"
].iloc[0]

improvement = baseline["MAE"] - hybrid["MAE"]
percentage = improvement / baseline["MAE"] * 100

selected_counts = (
    selection["Selected_Model"]
    .value_counts()
    .rename_axis("Selected Model")
    .reset_index(name="Market Series")
)

market_counts = market["Hybrid_Result"].value_counts()

rob = robustness.iloc[0]

def markdown_table(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)

comparison_table = comparison[
    ["Model", "Records", "MAE", "RMSE",
     "Median_Absolute_Error", "P90_Absolute_Error", "R2"]
].copy()

selection_table = selected_counts

report = f"""# Onion Price Prediction – Final Model Evaluation

## 1. Project overview

This experiment evaluates historical onion modal-price predictions across
multiple agricultural markets in India.

The project compares machine-learning regression models against a
Previous Price Baseline and evaluates a validation-driven hybrid strategy.

This is a historical prediction experiment, not a live market-price service.

## 2. Dataset and temporal evaluation

- Prediction records: {len(predictions):,}
- Market series: {market.shape[0]:,}
- Test period: {start_date} to {end_date}
- Validation period: September 2025
- Training period: January to August 2025 for validation model selection;
  the original full-training models used January to September 2025.
- Target: Modal price per quintal.
- Evaluation: chronological holdout.

The prediction task uses the previous observed price and other historical
features to predict the next available observation in each market series.
It should not be interpreted as a guaranteed next-calendar-day forecast.

## 3. Models evaluated

- Previous Price Baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Hybrid Strategy

The hybrid strategy selects a model using September validation MAE for
each eligible market series. Where no validation selection is available,
the baseline is used.

## 4. Final test comparison

{markdown_table(comparison_table)}

The hybrid strategy achieved an MAE of ₹{hybrid['MAE']:.2f}, compared with
₹{baseline['MAE']:.2f} for the Previous Price Baseline.

Observed MAE reduction: ₹{improvement:.2f} per prediction
(approximately {percentage:.2f}%).

The hybrid also achieved an RMSE of ₹{hybrid['RMSE']:.2f}, compared with
₹{baseline['RMSE']:.2f} for the baseline.

## 5. Validation-based model selection

{markdown_table(selection_table)}

There were {len(selection):,} market series with validation-based
model selection. The test set contains {market.shape[0]:,} market series.
Any additional series without a validation selection use the baseline
fallback.

## 6. Market-level comparison

- Hybrid wins: {int(market_counts.get('Hybrid Wins', 0))}
- Baseline wins: {int(market_counts.get('Baseline Wins', 0))}
- Ties: {int(market_counts.get('Tie', 0))}

The hybrid strategy does not improve every market. Some markets benefit
from machine learning, while others perform better with the baseline.

## 7. Bootstrap robustness analysis

- Bootstrap method: resampling complete market series.
- Bootstrap iterations: {int(rob['Bootstrap_Iterations']):,}
- Observed record-weighted MAE improvement: ₹{rob['Weighted_MAE_Improvement']:.4f}
- 95% bootstrap interval: ₹{rob['Bootstrap_95_Lower']:.4f}
  to ₹{rob['Bootstrap_95_Upper']:.4f}

The interval includes zero. Therefore, the observed improvement is not
clearly distinguishable from zero using this bootstrap analysis.

The hybrid should be described as showing a small observed improvement,
not as statistically proven to be superior.

## 8. Limitations

1. The hybrid improvement is small.
2. The 95% bootstrap interval includes zero.
3. Performance varies across individual market series.
4. Market observations are intermittent and do not always represent
   consecutive calendar days.
5. The available onion dataset's latest observation is April 20, 2026.
   It cannot support claims about live November 2026 prices.
6. Historical test performance does not guarantee future performance.
7. Data cleaning includes normalization of reported minimum and maximum
   prices around modal prices. This assumption must be disclosed when
   describing data preparation.

## 9. Final conclusion

The Previous Price Baseline is a strong benchmark for onion price
prediction.

The validation-driven hybrid strategy achieved a slightly lower test MAE
than the baseline. However, the bootstrap confidence interval includes
zero, and market-level results are mixed.

The experiment demonstrates the importance of chronological validation,
baseline comparison, market-level evaluation and robustness analysis.

Further testing on newer, independently collected data is required before
recommending the hybrid strategy for operational use.

## 10. Generated supporting reports

- onion_hybrid_test_comparison.csv
- onion_hybrid_test_predictions.csv
- onion_hybrid_validation_selection.csv
- onion_hybrid_market_improvement_analysis.csv
- onion_hybrid_robustness_analysis.csv
"""

REPORT.write_text(report, encoding="utf-8")

print("FINAL EVALUATION REPORT CREATED")
print("Location:", REPORT)
print("Test records:", len(predictions))
print("Market series:", len(market))
print("Hybrid MAE:", round(hybrid["MAE"], 2))
print("Baseline MAE:", round(baseline["MAE"], 2))
print("MAE improvement:", round(improvement, 2))
print("\nReport size:", REPORT.stat().st_size, "bytes")
