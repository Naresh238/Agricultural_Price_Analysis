from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "onion_dataset_deduplicated_v2.csv"
)

SELECTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "onion_hybrid_validation_selection.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "onion_hybrid_test_predictions.csv"
)

GROUP_COLUMNS = [
    "State",
    "District",
    "Market",
    "Variety",
    "Grade",
]

PREDICTION_COLUMNS = [
    "State",
    "District",
    "Market",
    "Variety",
    "Grade",
    "Arrival_Date",
    "Actual_Price",
    "Predicted_Price",
    "Model",
    "Selected_Model",
    "Absolute_Error",
]


def load_hybrid_results():
    """Load and validate the historical hybrid test results."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid predictions not found: {PREDICTIONS_PATH}"
        )

    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Validation selection report not found: {SELECTION_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    selection = pd.read_csv(SELECTION_PATH)

    missing_predictions = (
        set(PREDICTION_COLUMNS) - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(sorted(missing_predictions))
        )

    missing_selection = (
        set(GROUP_COLUMNS + ["Selected_Model"])
        - set(selection.columns)
    )

    if missing_selection:
        raise ValueError(
            "Missing selection columns: "
            + ", ".join(sorted(missing_selection))
        )

    predictions["Arrival_Date"] = pd.to_datetime(
        predictions["Arrival_Date"],
        errors="coerce",
    )

    for column in [
        "Actual_Price",
        "Predicted_Price",
        "Absolute_Error",
    ]:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    predictions = predictions.dropna(
        subset=[
            "Arrival_Date",
            "Actual_Price",
            "Predicted_Price",
            "Absolute_Error",
        ]
    ).copy()

    selection = selection.drop_duplicates(
        subset=GROUP_COLUMNS
    ).copy()

    return predictions, selection


def get_market_options(predictions):
    """Return the unique market-series choices in the test report."""

    return (
        predictions[GROUP_COLUMNS]
        .drop_duplicates()
        .sort_values(GROUP_COLUMNS)
        .reset_index(drop=True)
    )


def get_market_predictions(
    predictions,
    state,
    district,
    market,
    variety,
    grade,
):
    """Return historical test predictions for one market series."""

    values = {
        "State": state,
        "District": district,
        "Market": market,
        "Variety": variety,
        "Grade": grade,
    }

    result = predictions.copy()

    for column, value in values.items():
        result = result[result[column] == value]

    return (
        result.sort_values("Arrival_Date")
        .reset_index(drop=True)
    )


def get_selected_model(
    selection,
    state,
    district,
    market,
    variety,
    grade,
):
    """Return the model chosen using the validation period."""

    values = {
        "State": state,
        "District": district,
        "Market": market,
        "Variety": variety,
        "Grade": grade,
    }

    result = selection.copy()

    for column, value in values.items():
        result = result[result[column] == value]

    if result.empty:
        return "Previous Price Baseline"

    return str(result.iloc[0]["Selected_Model"])


def get_data_freshness():
    """Return the latest date in the source onion history."""

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Onion history not found: {HISTORY_PATH}"
        )

    dates = pd.read_csv(
        HISTORY_PATH,
        usecols=["Arrival_Date"],
    )

    dates["Arrival_Date"] = pd.to_datetime(
        dates["Arrival_Date"],
        errors="coerce",
    )

    return dates["Arrival_Date"].max()


if __name__ == "__main__":
    predictions, selection = load_hybrid_results()
    options = get_market_options(predictions)

    print("\n========== ONION HYBRID PREDICTOR TEST ==========")
    print("Prediction records:", len(predictions))
    print("Market series:", len(options))
    print("Validation selections:", len(selection))

    print("\nTest date range:")
    print(
        predictions["Arrival_Date"].min(),
        "to",
        predictions["Arrival_Date"].max(),
    )

    print("\nLatest source history:")
    print(get_data_freshness())

    print("\nFirst five market choices:")
    print(options.head().to_string(index=False))

    example = options.iloc[0]

    market_results = get_market_predictions(
        predictions,
        example["State"],
        example["District"],
        example["Market"],
        example["Variety"],
        example["Grade"],
    )

    selected_model = get_selected_model(
        selection,
        example["State"],
        example["District"],
        example["Market"],
        example["Variety"],
        example["Grade"],
    )

    print("\nExample market:")
    print(example.to_dict())

    print("\nSelected model:", selected_model)

    print("\nHistorical predictions:")
    print(
        market_results[
            [
                "Arrival_Date",
                "Actual_Price",
                "Predicted_Price",
                "Selected_Model",
                "Absolute_Error",
            ]
        ].head(5).to_string(index=False)
    )

    print("\nPREDICTOR MODULE TEST PASSED")
