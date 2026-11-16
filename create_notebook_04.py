import nbformat
from pathlib import Path

OUTPUT = Path("notebooks/04_Feature_Engineering.ipynb")

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell("""# 04 – Feature Engineering

## Agricultural Price Analysis and Onion Price Prediction

**Objective:** Understand and validate the historical price and date features used by the onion price prediction models.

This notebook uses the existing `onion_modeling_dataset_v1.csv`. It does not regenerate or overwrite the dataset.

The target variable is `Modal_Price`.
"""))

cells.append(nbformat.v4.new_markdown_cell("""## 1. Import Libraries

We use pandas for data analysis, NumPy for numerical validation, and pathlib for file management."""))

cells.append(nbformat.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd().resolve()

for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "data" / "processed" / "onion_modeling_dataset_v1.csv").is_file():
        PROJECT_ROOT = candidate
        break

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "onion_modeling_dataset_v1.csv"
SCRIPT_FILE = PROJECT_ROOT / "src" / "prepare_onion_modeling_dataset.py"

print("Project root:", PROJECT_ROOT)
print("Dataset exists:", DATA_FILE.is_file())
print("Feature engineering script exists:", SCRIPT_FILE.is_file())"""))

cells.append(nbformat.v4.new_markdown_cell("""## 2. Load the Existing Modeling Dataset

The modeling dataset was prepared earlier using the deduplicated onion price records and eligible market series."""))

cells.append(nbformat.v4.new_code_cell("""if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Modeling dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("Shape:", df.shape)

display(df.head())"""))

cells.append(nbformat.v4.new_markdown_cell("""## 3. Identify Feature Columns

The model uses market identity columns, historical price features, and calendar features.

`Modal_Price` is the prediction target. `Dataset_Split` identifies training and testing records."""))

cells.append(nbformat.v4.new_code_cell("""group_columns = [
    "State", "District", "Market", "Variety", "Grade"
]

numeric_features = [
    "Lag_1", "Lag_2",
    "Rolling_Mean_7", "Rolling_Mean_30",
    "Previous_Price_Change",
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear"
]

target = "Modal_Price"

required_columns = (
    group_columns
    + ["Arrival_Date", target, "Dataset_Split"]
    + numeric_features
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

print("Required columns verified.")
print("Model features:", numeric_features)
print("Target:", target)
print("Market grouping columns:", group_columns)"""))

cells.append(nbformat.v4.new_markdown_cell("""## 4. Understand Historical Price Features

- **Lag_1:** Previous available modal price for the same market series.
- **Lag_2:** Modal price from two previous available observations.
- **Rolling_Mean_7:** Average of up to seven previous available prices.
- **Rolling_Mean_30:** Average of up to thirty previous available prices.
- **Previous_Price_Change:** Difference between the two previous prices (`Lag_1 - Lag_2`).

The rolling calculations use `shift(1)` before calculating the rolling mean. This is important because the current target price must not be included in its own input features."""))

cells.append(nbformat.v4.new_code_cell("""display(
    df[
        group_columns
        + ["Arrival_Date", "Lag_1", "Lag_2",
           "Rolling_Mean_7", "Rolling_Mean_30",
           "Previous_Price_Change", target]
    ].head(10)
)"""))

cells.append(nbformat.v4.new_markdown_cell("""## 5. Validate Historical Price Features

Check whether the previous price change matches the difference between the two lag features.

This is a consistency check on the saved dataset, not a complete proof of model performance."""))

cells.append(nbformat.v4.new_code_cell("""expected_change = df["Lag_1"] - df["Lag_2"]

change_matches = np.isclose(
    df["Previous_Price_Change"].to_numpy(dtype=float),
    expected_change.to_numpy(dtype=float),
    equal_nan=True
)

print("Previous price change consistency:", bool(change_matches.all()))
print("Rows checked:", f"{len(df):,}")

if not change_matches.all():
    raise ValueError("Previous_Price_Change validation failed.")"""))

cells.append(nbformat.v4.new_markdown_cell("""## 6. Calendar Feature Engineering

Calendar features help the model represent seasonal and time-related patterns.

- **Year:** Calendar year.
- **Month:** Month number (1–12).
- **Day:** Day of the month.
- **DayOfWeek:** Monday = 0 through Sunday = 6.
- **DayOfYear:** Day number within the year."""))

cells.append(nbformat.v4.new_code_cell("""calendar_checks = {
    "Year": df["Arrival_Date"].dt.year,
    "Month": df["Arrival_Date"].dt.month,
    "Day": df["Arrival_Date"].dt.day,
    "DayOfWeek": df["Arrival_Date"].dt.dayofweek,
    "DayOfYear": df["Arrival_Date"].dt.dayofyear
}

for feature, expected in calendar_checks.items():
    matches = df[feature].eq(expected).all()
    print(f"{feature} matches Arrival_Date:", bool(matches))
    if not matches:
        raise ValueError(f"Calendar feature validation failed: {feature}")

print("\\nAll calendar features validated.")"""))

cells.append(nbformat.v4.new_markdown_cell("""## 7. Check Missing Values and Feature Types

Review missing values and data types before model training."""))

cells.append(nbformat.v4.new_code_cell("""print("Missing values in model features:")
display(df[numeric_features].isna().sum().to_frame("Missing_Count"))

print("\\nFeature data types:")
display(df[numeric_features].dtypes.to_frame("Data_Type"))

print("\\nTarget summary:")
display(df[target].describe().to_frame("Modal_Price"))"""))

cells.append(nbformat.v4.new_markdown_cell("""## 8. Review the Temporal Train-Test Split

The existing workflow uses records from January 3 through September 30, 2025 for training and October 1 through November 5, 2025 for testing.

The split is chronological rather than random. The model predicts the next available observation for each eligible market series."""))

cells.append(nbformat.v4.new_code_cell("""train = df[df["Dataset_Split"] == "Train"].copy()
test = df[df["Dataset_Split"] == "Test"].copy()

print("Training records:", f"{len(train):,}")
print("Testing records:", f"{len(test):,}")

print("\\nTraining date range:")
print(train["Arrival_Date"].min(), "to", train["Arrival_Date"].max())

print("\\nTesting date range:")
print(test["Arrival_Date"].min(), "to", test["Arrival_Date"].max())

print("\\nTraining market series:", train[group_columns].drop_duplicates().shape[0])
print("Testing market series:", test[group_columns].drop_duplicates().shape[0])

if not train.empty and not test.empty:
    assert train["Arrival_Date"].max() < test["Arrival_Date"].min()
    print("\\nChronological split verified.")"""))

cells.append(nbformat.v4.new_markdown_cell("""## 9. Inspect the Feature Engineering Source Code

The source script is the implementation used to create the existing modeling dataset. Inspect its rolling-feature logic to confirm that previous observations are used."""))

cells.append(nbformat.v4.new_code_cell("""if not SCRIPT_FILE.is_file():
    raise FileNotFoundError(f"Source script not found: {SCRIPT_FILE}")

script_text = SCRIPT_FILE.read_text(encoding="utf-8")

checks = {
    "Lag features use group shift": 'grouped.shift(1)' in script_text,
    "Second lag uses group shift": 'grouped.shift(2)' in script_text,
    "Seven-observation rolling mean uses shift": 's.shift(1).rolling(7' in script_text,
    "Thirty-observation rolling mean uses shift": 's.shift(1).rolling(30' in script_text
}

for description, passed in checks.items():
    print(description + ":", "PASS" if passed else "CHECK")

if not all(checks.values()):
    raise ValueError("Review the feature engineering source code.")

print("\\nHistorical feature logic verified in source script.")"""))

cells.append(nbformat.v4.new_markdown_cell("""## 10. Feature Engineering Summary

The existing modeling dataset contains historical price features and calendar features.

### Important methodology notes

1. Historical features are calculated separately for each State, District, Market, Variety and Grade series.
2. The rolling means use prior observations through `shift(1)`.
3. The target variable is `Modal_Price`.
4. The temporal test period follows the training period.
5. The next available market observation is predicted; observations are not necessarily consecutive calendar days.
6. This notebook validates the saved dataset and source logic. It does not retrain models or overwrite existing files.

### Next notebook

`05_Model_Training.ipynb` – compare machine learning algorithms using the prepared features."""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3"
    }
}

nbformat.validate(nb)
nbformat.write(nb, OUTPUT)

print("NOTEBOOK 04 CREATED SUCCESSFULLY")
print("File:", OUTPUT)
print("Cells:", len(nb.cells))
print("Markdown:", sum(c.cell_type == "markdown" for c in nb.cells))
print("Code:", sum(c.cell_type == "code" for c in nb.cells))
print("Size:", f"{OUTPUT.stat().st_size:,}", "bytes")
