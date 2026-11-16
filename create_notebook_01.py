from pathlib import Path
import nbformat

output = Path("notebooks/01_Data_Collection.ipynb")

if output.exists() and output.stat().st_size > 0:
    raise FileExistsError(f"{output} is not empty. Keeping it safe.")

output.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell(r"""
# 01 – Agricultural Price Data Collection

## Project: Agricultural Price Analysis and Onion Price Prediction

**Objective:** Understand the raw agricultural price datasets, inspect their structures, document the data-combination process, and identify the onion dataset used for subsequent analysis.

### Data sources

1. `agricultural_prices_2025.csv` – daily agricultural market prices for 2025.
2. `agricultural_prices_2026.csv` – daily agricultural market prices for 2026.
3. `agricultural_prices_historical_monthly.csv` – separate historical monthly dataset covering Maharashtra during 2015–2016.

**Important:** The historical monthly dataset has a different time frequency and is not merged into the 2025–2026 daily dataset.
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 1. Import required libraries

We use pandas for CSV inspection and pathlib for safe file-path handling.

This notebook inspects existing files. It does not download data, overwrite datasets, or retrain models.
"""))

cells.append(nbformat.v4.new_code_cell("""
from pathlib import Path
import pandas as pd
import os

PROJECT_ROOT = Path.cwd()

if not (PROJECT_ROOT / "data").exists():
    raise FileNotFoundError(
        "Run this notebook from the Agricultural_Price_Analysis project root."
    )

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("Project directory:", PROJECT_ROOT)
print("Raw data directory:", RAW_DIR)
print("Processed data directory:", PROCESSED_DIR)
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 2. Inspect raw data files

The first step is to verify that the expected source files exist and inspect their sizes.

The files are large, so this step uses file metadata rather than loading entire datasets into memory.
"""))

cells.append(nbformat.v4.new_code_cell("""
raw_files = [
    "agricultural_prices_2025.csv",
    "agricultural_prices_2026.csv",
    "agricultural_prices_historical_monthly.csv"
]

for filename in raw_files:
    path = RAW_DIR / filename

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{filename}: {size_mb:,.2f} MB")
    else:
        print(f"MISSING: {filename}")
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 3. Inspect source dataset columns

Read only the first five rows to inspect the structure without loading the full datasets.

The daily datasets are expected to contain:

- State
- District
- Market
- Commodity
- Variety
- Grade
- Arrival_Date
- Min_Price
- Max_Price
- Modal_Price
- Commodity_Code
"""))

cells.append(nbformat.v4.new_code_cell("""
daily_2025 = RAW_DIR / "agricultural_prices_2025.csv"
daily_2026 = RAW_DIR / "agricultural_prices_2026.csv"
historical = RAW_DIR / "agricultural_prices_historical_monthly.csv"

for path in [daily_2025, daily_2026, historical]:
    print("\\n" + "=" * 65)
    print("FILE:", path.name)

    if path.exists():
        sample = pd.read_csv(path, nrows=5)
        print("Columns:", sample.columns.tolist())
        print("Sample shape:", sample.shape)
        display(sample.head())
    else:
        print("File not found.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 4. Document the safe dataset combination process

The project uses `src/combine_datasets_safe.py` to combine the 2025 and 2026 daily datasets.

The script:

1. Reads both files in chunks of 100,000 rows.
2. Checks that the expected columns are present.
3. Converts `Arrival_Date` to datetime.
4. Removes invalid dates.
5. Excludes records dated 30 December 2025 from the 2026 source to avoid the known date overlap, retaining that date from the 2025 source.
6. Removes records missing State, District, Market, or Commodity.
7. Writes the combined data to:

`data/processed/agricultural_prices_combined_v2.csv`

The separate historical monthly dataset is not included in this combination.
"""))

cells.append(nbformat.v4.new_code_cell("""
combined_file = PROCESSED_DIR / "agricultural_prices_combined_v2.csv"

if combined_file.exists():
    print("Combined dataset exists.")
    print("File size (MB):", round(combined_file.stat().st_size / (1024 * 1024), 2))
else:
    print("Combined dataset not found:", combined_file)

print("\\nCombination script:")
print(PROJECT_ROOT / "src" / "combine_datasets_safe.py")
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 5. Document onion data extraction

The script `src/extract_onion_dataset.py` reads the cleaned combined dataset in chunks of 200,000 rows.

It selects records where the Commodity value is Onion, ignoring differences in letter case and surrounding spaces.

The extracted data is saved to:

`data/processed/onion_dataset_v2.csv`

The extraction script protects an existing output file and stops instead of overwriting it.
"""))

cells.append(nbformat.v4.new_code_cell("""
cleaned_file = PROCESSED_DIR / "agricultural_prices_cleaned_v2.csv"
onion_file = PROCESSED_DIR / "onion_dataset_v2.csv"

for path in [cleaned_file, onion_file]:
    print("\\nFILE:", path)

    if path.exists():
        print("Exists: Yes")
        print("Size (MB):", round(path.stat().st_size / (1024 * 1024), 2))
    else:
        print("Exists: No")
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 6. Inspect the extracted onion dataset

Read a small sample to verify its columns and inspect a few observations.

This cell does not load the entire onion dataset.
"""))

cells.append(nbformat.v4.new_code_cell("""
if onion_file.exists():
    onion_sample = pd.read_csv(onion_file, nrows=10)

    print("Columns:")
    print(onion_sample.columns.tolist())

    print("\\nSample records:")
    display(onion_sample)

    if "Commodity" in onion_sample.columns:
        print("\\nCommodity values in sample:")
        print(onion_sample["Commodity"].value_counts(dropna=False))
else:
    print("Onion dataset not found. Complete the project data preparation first.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""
## 7. Data collection workflow summary

The main daily-data workflow is:

**2025 and 2026 raw datasets → Safe combination → Data cleaning → Onion extraction → Onion analysis dataset**

The historical monthly dataset is retained as a separate source.

### Data handling considerations

- Large files are processed in chunks to reduce memory usage.
- The known overlapping date is handled during combination.
- Invalid dates and records missing essential identification fields are excluded by the combination script.
- Existing output files are protected from accidental overwriting.
- Exact duplicate removal and additional quality checks are handled in later preprocessing steps.

## 8. Conclusion

The project uses daily agricultural market data from 2025 and 2026 to construct an onion-specific dataset for multi-market price analysis.

The separate 2015–2016 historical monthly dataset is documented independently. The extracted onion dataset is the input for the subsequent preprocessing, exploratory analysis, feature engineering, model training, and evaluation stages.

**Next notebook:** `02_Data_Preprocessing.ipynb`
"""))

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
nbformat.write(nb, output)

print("NOTEBOOK CREATED SUCCESSFULLY")
print("File:", output)
print("Cells:", len(nb.cells))
print("Size:", output.stat().st_size, "bytes")
