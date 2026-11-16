# Onion Price Prediction – Final Model Evaluation

## 1. Project overview

This experiment evaluates historical onion modal-price predictions across
multiple agricultural markets in India.

The project compares machine-learning regression models against a
Previous Price Baseline and evaluates a validation-driven hybrid strategy.

This is a historical prediction experiment, not a live market-price service.

## 2. Dataset and temporal evaluation

- Prediction records: 12,978
- Market series: 751
- Test period: 2025-10-01 to 2025-11-05
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

| Model | Records | MAE | RMSE | Median_Absolute_Error | P90_Absolute_Error | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid Strategy | 12978 | 89.59 | 241.56 | 12.54 | 250.00 | 0.94 |
| Previous Price Baseline | 12978 | 90.21 | 252.93 | 5.00 | 280.00 | 0.93 |
| Linear Regression | 12978 | 99.53 | 215.07 | 35.20 | 252.82 | 0.95 |
| Gradient Boosting | 12978 | 101.17 | 214.91 | 40.03 | 257.45 | 0.95 |
| XGBoost | 12978 | 101.77 | 235.24 | 30.29 | 258.73 | 0.94 |
| Random Forest | 12978 | 104.61 | 218.77 | 36.34 | 277.48 | 0.95 |

The hybrid strategy achieved an MAE of ₹89.59, compared with
₹90.21 for the Previous Price Baseline.

Observed MAE reduction: ₹0.61 per prediction
(approximately 0.68%).

The hybrid also achieved an RMSE of ₹241.56, compared with
₹252.93 for the baseline.

## 5. Validation-based model selection

| Selected Model | Market Series |
| --- | --- |
| Previous Price Baseline | 460 |
| Linear Regression | 155 |
| XGBoost | 75 |
| Gradient Boosting | 60 |

There were 750 market series with validation-based
model selection. The test set contains 751 market series.
Any additional series without a validation selection use the baseline
fallback.

## 6. Market-level comparison

- Hybrid wins: 154
- Baseline wins: 136
- Ties: 461

The hybrid strategy does not improve every market. Some markets benefit
from machine learning, while others perform better with the baseline.

## 7. Bootstrap robustness analysis

- Bootstrap method: resampling complete market series.
- Bootstrap iterations: 10,000
- Observed record-weighted MAE improvement: ₹0.6133
- 95% bootstrap interval: ₹-1.7306
  to ₹2.6022

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
