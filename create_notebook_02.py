import nbformat
from pathlib import Path

OUTPUT = Path("notebooks/02_Data_Preprocessing.ipynb")

if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
    raise FileExistsError(
        f"{OUTPUT} already contains data. Keeping it safe."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python"}
}

cells = [
    nbformat.v4.new_markdown_cell("""# 02 – Agricultural Data Preprocessing

## Project: Agricultural Price Analysis and Onion Price Prediction

### Objective
This notebook documents the preprocessing workflow used to prepare agricultural price records and create a reliable onion dataset for exploratory data analysis and machine learning.

### Main activities
1. Inspect existing processed datasets.
2. Understand agricultural price cleaning rules.
3. Review the onion extraction output.
4. Examine exact duplicate removal.
5. Validate the final onion dataset.

**Important:** This notebook inspects existing project outputs. It does not rerun the original cleaning, extraction, or deduplication scripts."""),
    nbformat.v4.new_markdown_cell("""## 1. Import Libraries

We use:
- pandas for data inspection and validation.
- pathlib for safe file path handling.
- Python's built-in functions for file information."""),
    nbformat.v4.new_code_cell("""from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

print("Project directory:", PROJECT_ROOT)
print("Pandas version:", pd.__version__)"""),
    nbformat.v4.new_markdown_cell("""## 2. Inspect Existing Preprocessing Files

The project keeps raw data separate from processed data.

Important outputs:
- `agricultural_prices_combined_v2.csv`
- `agricultural_prices_cleaned_v2.csv`
- `onion_dataset_v2.csv`
- `onion_dataset_deduplicated_v2.csv`

The following cell checks whether these files exist and reports their sizes. It does not modify them."""),
    nbformat.v4.new_code_cell("""files = [
    Path("data/processed/agricultural_prices_combined_v2.csv"),
    Path("data/processed/agricultural_prices_cleaned_v2.csv"),
    Path("data/processed/onion_dataset_v2.csv"),
    Path("data/processed/onion_dataset_deduplicated_v2.csv")
]

for file in files:
    if file.exists():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"{file}: {size_mb:,.2f} MB")
    else:
        print(f"MISSING: {file}")"""),
    nbformat.v4.new_markdown_cell("""## 3. Agricultural Price Cleaning

The cleaning script `src/clean_agricultural_data_v2.py` processes the combined dataset in chunks of 200,000 rows.

### Cleaning operations
1. Convert `Min_Price`, `Max_Price`, and `Modal_Price` to numeric values.
2. Remove rows with missing modal prices.
3. Keep rows where modal price is positive and minimum/maximum prices are non-negative.
4. Replace zero minimum or maximum prices with the modal price.
5. Adjust the minimum and maximum so the modal price falls within the recorded range.

### Important methodological note
Replacing zero values and adjusting the price range are project-specific data normalization assumptions. They do not prove that the original source values were incorrect. The original modal price is preserved as the prediction target.

The script writes `data/processed/agricultural_prices_cleaned_v2.csv` and protects an existing output from accidental overwrite."""),
    nbformat.v4.new_markdown_cell("""## 4. Inspect a Sample of the Cleaned Agricultural Dataset

Only a small sample is loaded here. The complete multi-million-row dataset is not loaded into memory."""),
    nbformat.v4.new_code_cell("""cleaned_file = Path("data/processed/agricultural_prices_cleaned_v2.csv")

if cleaned_file.exists():
    sample = pd.read_csv(cleaned_file, nrows=5)
    print("Sample shape:", sample.shape)
    display(sample)
    print("\\nColumns:")
    print(sample.columns.tolist())
else:
    print("Cleaned agricultural dataset was not found.")"""),
    nbformat.v4.new_markdown_cell("""## 5. Onion Dataset Extraction

The script `src/extract_onion_dataset.py` reads the cleaned agricultural dataset in chunks of 200,000 rows.

It selects records whose `Commodity` value is Onion, using case-insensitive matching after trimming whitespace.

Output:
`data/processed/onion_dataset_v2.csv`

The extraction script protects an existing output file and is not rerun in this notebook."""),
    nbformat.v4.new_code_cell("""onion_file = Path("data/processed/onion_dataset_v2.csv")

if onion_file.exists():
    onion_sample = pd.read_csv(onion_file, nrows=5)
    print("Onion dataset sample:")
    display(onion_sample)
    print("\\nColumns:")
    print(onion_sample.columns.tolist())
else:
    print("Onion dataset was not found.")"""),
    nbformat.v4.new_markdown_cell("""## 6. Initial Onion Data Quality Checks

The original quality-check script examines:
- Missing values.
- Exact duplicate rows.
- Duplicate market-date keys.
- Invalid arrival dates.
- Negative or zero modal prices.
- Modal prices outside the minimum/maximum range.
- Unique markets, districts, and states.
- Records by year.

**Important distinction:** A duplicate market-date key is not necessarily an exact duplicate row. The deduplication script removes exact duplicate rows only."""),
    nbformat.v4.new_code_cell("""if onion_file.exists():
    original = pd.read_csv(onion_file)
    original["Arrival_Date"] = pd.to_datetime(
        original["Arrival_Date"], errors="coerce"
    )

    key_columns = [
        "State", "District", "Market",
        "Commodity", "Variety", "Grade", "Arrival_Date"
    ]

    print("Original onion records:", f"{len(original):,}")
    print("Missing values by column:")
    display(original.isnull().sum().to_frame("Missing values"))

    print("Exact duplicate rows:", f"{original.duplicated().sum():,}")
    print(
        "Duplicate market-date keys:",
        f"{original.duplicated(subset=key_columns).sum():,}"
    )
    print("Invalid dates:", f"{original['Arrival_Date'].isna().sum():,}")
    print("Negative modal prices:", f"{(original['Modal_Price'] < 0).sum():,}")
    print("Zero modal prices:", f"{(original['Modal_Price'] == 0).sum():,}")
    print("Unique markets:", f"{original['Market'].nunique():,}")
    print("Unique districts:", f"{original['District'].nunique():,}")
    print("Unique states:", f"{original['State'].nunique():,}")

    print("\\nRecords by year:")
    display(original["Arrival_Date"].dt.year.value_counts().sort_index())
else:
    print("Original onion dataset was not found.")"""),
    nbformat.v4.new_markdown_cell("""## 7. Exact Duplicate Removal

The script `src/deduplicate_onion.py` uses pandas `drop_duplicates(keep="first")`.

It removes rows that are identical across all columns and preserves the first occurrence.

### Recorded project result
- Original onion dataset: 241,273 records.
- Exact duplicate rows removed: 89.
- Deduplicated dataset: 241,184 records.

The original dataset is preserved. The deduplicated output is saved separately as `onion_dataset_deduplicated_v2.csv`.

This operation does not remove every repeated market-date key; it removes exact full-row duplicates only."""),
    nbformat.v4.new_code_cell("""deduplicated_file = Path(
    "data/processed/onion_dataset_deduplicated_v2.csv"
)

if onion_file.exists() and deduplicated_file.exists():
    original_count = sum(
        len(chunk) for chunk in pd.read_csv(onion_file, chunksize=50000)
    )
    final_count = sum(
        len(chunk) for chunk in pd.read_csv(deduplicated_file, chunksize=50000)
    )

    print("Original onion records:", f"{original_count:,}")
    print("Deduplicated onion records:", f"{final_count:,}")
    print("Records removed:", f"{original_count - final_count:,}")
else:
    print("One or both onion datasets are missing.")"""),
    nbformat.v4.new_markdown_cell("""## 8. Final Deduplicated Dataset Validation

The validation script `src/validate_onion_deduplicated.py` checks the final dataset for:
- Total number of records.
- Remaining exact duplicates.
- Unique states and markets.
- Invalid arrival dates.
- Start and end dates.
- Records by year.

The next cell reads the final onion dataset for validation only."""),
    nbformat.v4.new_code_cell("""if deduplicated_file.exists():
    final_df = pd.read_csv(deduplicated_file)

    final_df["Arrival_Date"] = pd.to_datetime(
        final_df["Arrival_Date"], errors="coerce"
    )

    print("Total records:", f"{len(final_df):,}")
    print("Exact duplicate rows:", f"{final_df.duplicated().sum():,}")
    print("Unique states:", f"{final_df['State'].nunique():,}")
    print("Unique markets:", f"{final_df['Market'].nunique():,}")
    print("Invalid dates:", f"{final_df['Arrival_Date'].isna().sum():,}")

    valid_dates = final_df["Arrival_Date"].dropna()
    if not valid_dates.empty:
        print("Start date:", valid_dates.min().date())
        print("End date:", valid_dates.max().date())

    print("\\nRecords by year:")
    display(final_df["Arrival_Date"].dt.year.value_counts().sort_index())

    print("\\nFinal dataset preview:")
    display(final_df.head())
else:
    print("Deduplicated onion dataset was not found.")"""),
    nbformat.v4.new_markdown_cell("""## 9. Preprocessing Summary

### Recorded results

| Stage | Records |
|---|---:|
| Extracted onion dataset | 241,273 |
| Exact duplicate rows removed | 89 |
| Final deduplicated onion dataset | 241,184 |

### Final validation results
- Exact duplicate rows: 0.
- Unique states: 28.
- Unique markets: 2,152.
- Date coverage: January 1, 2025 to April 20, 2026.

### Conclusion
The agricultural data was cleaned, onion records were extracted, and exact duplicate rows were removed. The final onion dataset contains 241,184 records and is ready for exploratory data analysis and subsequent feature engineering.

**Next notebook:** `03_EDA.ipynb` – Exploratory Data Analysis.""")
]

nb.cells = cells
nbformat.validate(nb)
nbformat.write(nb, OUTPUT)

print("NOTEBOOK CREATED SUCCESSFULLY")
print("File:", OUTPUT)
print("Cells:", len(nb.cells))
print("Markdown cells:", sum(c.cell_type == "markdown" for c in nb.cells))
print("Code cells:", sum(c.cell_type == "code" for c in nb.cells))
print("Size:", OUTPUT.stat().st_size, "bytes")
