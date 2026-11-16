from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = ROOT / "reports" / "onion_multi_market_test_predictions.csv"
COMPARISON_FILE = ROOT / "reports" / "onion_multi_market_model_comparison.csv"

EXPECTED_MODELS = {
    "Previous Price Baseline",
    "Linear Regression",
    "Gradient Boosting",
    "Random Forest",
    "XGBoost",
}

MODEL_FILES = [
    ROOT / "models" / "onion_multi_market_v1" / "linear_regression.joblib",
    ROOT / "models" / "onion_multi_market_v1" / "gradient_boosting.joblib",
    ROOT / "models" / "onion_multi_market_v1" / "random_forest.joblib",
    ROOT / "models" / "onion_multi_market_v1" / "xgboost.joblib",
]


@pytest.fixture(scope="module")
def predictions():
    assert PREDICTIONS_FILE.exists(), "Prediction CSV is missing."
    return pd.read_csv(PREDICTIONS_FILE)


@pytest.fixture(scope="module")
def comparison():
    assert COMPARISON_FILE.exists(), "Model comparison CSV is missing."
    return pd.read_csv(COMPARISON_FILE)


def test_prediction_file_has_required_columns(predictions):
    required = {
        "State", "District", "Market", "Variety", "Grade",
        "Arrival_Date", "Actual_Price", "Predicted_Price", "Model"
    }
    assert required.issubset(predictions.columns)


def test_all_expected_models_have_predictions(predictions):
    assert set(predictions["Model"].unique()) == EXPECTED_MODELS


def test_each_model_has_same_test_record_count(predictions):
    counts = predictions.groupby("Model").size()
    assert len(counts) == 5
    assert counts.nunique() == 1
    assert (counts > 0).all()


def test_prediction_values_are_valid(predictions):
    actual = pd.to_numeric(predictions["Actual_Price"], errors="coerce")
    predicted = pd.to_numeric(predictions["Predicted_Price"], errors="coerce")

    assert actual.notna().all(), "Actual prices contain missing or invalid values."
    assert predicted.notna().all(), "Predictions contain missing or invalid values."
    assert np.isfinite(actual).all()
    assert np.isfinite(predicted).all()
    assert (actual >= 0).all()


def test_comparison_report_has_all_models(comparison):
    required = {"Model", "MAE", "RMSE", "R2", "Training_Seconds", "Status"}
    assert required.issubset(comparison.columns)
    assert set(comparison["Model"]) == EXPECTED_MODELS
    assert (comparison["Status"] == "Success").all()


def test_recalculated_metrics_match_saved_report(predictions, comparison):
    for model_name in EXPECTED_MODELS:
        group = predictions[predictions["Model"] == model_name]

        actual = group["Actual_Price"].to_numpy()
        predicted = group["Predicted_Price"].to_numpy()

        recalculated = {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        }

        saved = comparison[comparison["Model"] == model_name].iloc[0]

        for metric, value in recalculated.items():
            assert abs(value - float(saved[metric])) < 0.02, (
                f"{model_name}: {metric} does not match the saved report."
            )


@pytest.mark.parametrize("model_file", MODEL_FILES)
def test_trained_model_file_exists(model_file):
    assert model_file.exists(), f"Missing trained model: {model_file.name}"
    assert model_file.stat().st_size > 0


@pytest.mark.parametrize(
    "filename",
    [
        "onion_model_evaluation_metrics.csv",
        "onion_model_evaluation_verification.csv",
        "onion_model_residual_summary.csv",
        "onion_model_state_evaluation.csv",
        "onion_model_largest_prediction_errors.csv",
    ],
)
def test_evaluation_report_exists(filename):
    report = ROOT / "reports" / filename
    assert report.exists(), f"Missing evaluation report: {filename}"
    assert report.stat().st_size > 0


@pytest.mark.parametrize(
    "filename",
    [
        "onion_model_evaluation_metrics.png",
        "onion_model_evaluation_residual_boxplot.png",
        "onion_actual_vs_predicted.png",
    ],
)
def test_evaluation_chart_exists(filename):
    chart = ROOT / "reports" / "figures" / filename
    assert chart.exists(), f"Missing evaluation chart: {filename}"
    assert chart.stat().st_size > 0

# ---------------------------------------------------------
# DATA DRIFT DETECTION TESTS
# ---------------------------------------------------------

DRIFT_REPORT = ROOT / "reports" / "drift_detection" / "train_vs_test_drift_report.csv"
DRIFT_CHART = ROOT / "reports" / "figures" / "onion_data_drift_psi.png"


def test_data_drift_report_exists():
    assert DRIFT_REPORT.exists(), "Data drift report is missing."
    assert DRIFT_REPORT.stat().st_size > 0


def test_data_drift_report_has_valid_results():
    assert DRIFT_REPORT.exists(), "Data drift report is missing."

    drift = pd.read_csv(DRIFT_REPORT)

    required = {
        "Feature",
        "Feature_Type",
        "PSI",
        "Reference_Records",
        "Comparison_Records",
        "Drift_Status",
    }

    assert required.issubset(drift.columns)
    assert len(drift) >= 1
    assert drift["Feature"].notna().all()
    assert pd.to_numeric(drift["PSI"], errors="coerce").notna().all()
    assert (drift["PSI"] >= 0).all()

    allowed_statuses = {"Low", "Moderate", "High"}
    assert set(drift["Drift_Status"]).issubset(allowed_statuses)

    assert (drift["Reference_Records"] > 0).all()
    assert (drift["Comparison_Records"] > 0).all()


def test_data_drift_chart_exists():
    assert DRIFT_CHART.exists(), "Data drift chart is missing."
    assert DRIFT_CHART.stat().st_size > 0
