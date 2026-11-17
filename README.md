# Agricultural Price Analysis and Prediction Using Machine Learning and Tableau

## 1. Project Overview

This project analyzes agricultural market prices and develops machine learning models to predict onion modal prices across Indian agricultural markets.

The workflow includes data preprocessing, exploratory data analysis, feature engineering, machine learning, model evaluation, monitoring, data drift detection, automated testing, and interactive dashboards.

**Prediction Target:** Modal Price (₹/Quintal)

## 2. Technologies Used

- Python, Pandas and NumPy
- Scikit-learn and XGBoost
- Matplotlib and Plotly
- Streamlit
- Tableau
- Joblib
- Jupyter Notebook
- Pytest

## 3. Dataset Summary

The deduplicated onion dataset contains:

| Description | Value |
|---|---:|
| Onion records after deduplication | 241,184 |
| Market series | 3,717 |
| States | 28 |
| Districts | 416 |
| Markets | 2,152 |

### Modelling Dataset

**File:** `data/processed/onion_modeling_dataset_v1.csv`

| Description | Value |
|---|---:|
| Records | 168,526 |
| Columns | 22 |
| Training records | 155,548 |
| Testing records | 12,978 |
| Training period | 2025-01-03 to 2025-09-30 |
| Testing period | 2025-10-01 to 2025-11-05 |
| Commodity | Onion |

The deduplicated dataset summary and modelling dataset represent different stages of the data workflow.

## 4. Data Preprocessing and Feature Engineering

The workflow includes:
- Dataset inspection and data type verification.
- Missing-value and duplicate handling.
- Invalid price validation.
- Date conversion.
- Historical feature generation.
- Chronological train-test separation.

Engineered features include:
- Lag_1 and Lag_2
- Rolling_Mean_7 and Rolling_Mean_30
- Previous_Price_Change
- Year, Month, Day, DayOfWeek and DayOfYear

The prediction target is `Modal_Price`.

## 5. Agricultural Price Analysis

The project includes:
- Price distribution analysis.
- Market-wise price comparisons.
- Historical price trends.
- Geographical market analysis.
- Rolling average analysis.
- Prediction error analysis.
- Model performance comparison.

## 6. Machine Learning Models

Five approaches are evaluated:

1. Previous Price Baseline
2. Linear Regression
3. Random Forest Regression
4. Gradient Boosting Regression
5. XGBoost Regression

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Verified Model Performance

All models were evaluated on the same chronological test dataset containing 12,978 observations per model.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Previous Price Baseline | 90.2080 | 252.9331 | 0.929389 |
| Linear Regression | 99.5348 | 215.0746 | 0.948945 |
| Gradient Boosting | 101.1748 | 214.9149 | 0.949021 |
| Random Forest | 104.6054 | 218.7666 | 0.947177 |
| XGBoost | 101.7650 | 235.2395 | 0.938922 |

The Previous Price Baseline has the lowest MAE. Gradient Boosting has the lowest RMSE and highest R². No single approach is best across all three metrics.

## 7. Model Evaluation

Evaluation reports are stored in `reports/`:

- `onion_model_evaluation_metrics.csv`
- `onion_model_evaluation_verification.csv`
- `onion_model_residual_summary.csv`
- `onion_model_state_evaluation.csv`
- `onion_model_largest_prediction_errors.csv`

Evaluation charts include:

- `onion_model_evaluation_metrics.png`
- `onion_model_evaluation_residual_boxplot.png`
- `onion_actual_vs_predicted.png`

## 8. Model Monitoring

Monitoring outputs:

- `reports/monitoring/model_monitoring_summary.csv`
- `reports/monitoring/daily_model_performance.csv`
- `reports/monitoring/model_monitoring_history.csv`

The workflow calculates MAE, RMSE, R², mean residual, maximum absolute error, and daily model performance.

**Limitation:** Current monitoring uses existing historical test predictions. It is not continuous monitoring of newly arriving live market data.

## 9. Data Drift Detection

The project uses Population Stability Index (PSI) to compare training and testing distributions.

| Dataset | Records |
|---|---:|
| Training reference | 155,548 |
| Testing comparison | 12,978 |

Current report:
- 8 features have high PSI.
- 8 features have low PSI.
- No features have moderate PSI.

Month and DayOfYear have the largest PSI values, partly reflecting the chronological difference between training and testing periods. Historical price features and the Modal_Price target also show distribution changes requiring investigation.

PSI is a screening measure; high PSI alone does not prove model degradation.

Files:
- `reports/drift_detection/train_vs_test_drift_report.csv`
- `reports/figures/onion_data_drift_psi.png`

## 10. Automated Reliability Testing

Automated tests use Pytest to check prediction files, model outputs, evaluation metrics, model files, reports, charts, and drift detection outputs.

**Latest verified result: 21 tests passed.**

Run tests:

```powershell
python -m pytest .\tests -v --junitxml=.\reports\testing\reliability_test_results.xml
```

## 11. Streamlit Application

Run the application locally:

```powershell
streamlit run .\streamlit_app\app.py
```

Local address: `http://localhost:8501`

Application sections include:
- Dashboard
- Dataset Overview
- Commodity Analysis
- Price Prediction
- Model Performance
- Onion Hybrid Prediction

## 12. Tableau Dashboards

Four packaged Tableau workbooks are included in the `tableau/` directory:

1. `Agricultural Price overview (1).twbx` — Agricultural Price Overview
2. `Geographical Market price Analysis (1).twbx` — Geographical Market Price Analysis
3. `Dashboard-3.twbx` — Additional Tableau dashboard
4. `Dashboard-4.twbx` — Additional Tableau dashboard

Open these packaged workbooks using Tableau Desktop or a compatible Tableau application.

## 13. Project Structure

```text
Agricultural_Price_Analysis/
├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
├── reports/
│   ├── figures/
│   ├── monitoring/
│   ├── drift_detection/
│   └── testing/
├── streamlit_app/
├── tableau/
├── tests/
├── README.md
└── requirements.txt
```

## 14. Limitations

- The current forecasting experiment focuses on onion prices.
- Model performance may vary across markets and time periods.
- Distribution changes require further investigation.
- Monitoring currently uses historical test predictions.
- Continuous live data integration and public deployment are not yet established.

## 15. Future Enhancements

- Automate agricultural market data collection.
- Add more agricultural commodities.
- Improve forecasting and validation.
- Implement monitoring for newly arriving observations.
- Investigate data drift over time.
- Improve and deploy the Streamlit application.
- Publish and maintain Tableau dashboard access.

## Streamlit Dashboard Preview

The interactive Streamlit dashboard allows users to explore agricultural price trends and model predictions.

![Streamlit Dashboard](docs/images/streamlit_dashboard.png)

Run locally: streamlit run streamlit_app/app.py

