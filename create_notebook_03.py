import nbformat
from pathlib import Path

OUTPUT = Path("notebooks/03_EDA.ipynb")

if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
    raise FileExistsError(
        f"{OUTPUT} already exists. Keeping the existing notebook safe."
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
    nbformat.v4.new_markdown_cell("""# 03 – Exploratory Data Analysis (EDA)

## Project: Agricultural Price Analysis and Onion Price Prediction

### Objective
Explore the final deduplicated onion dataset to understand price patterns, geographic differences, market activity, and changes over time.

### Analysis covered
1. Dataset overview and statistical summary.
2. Data quality checks.
3. Onion modal price distribution.
4. Monthly and yearly price trends.
5. State-wise price comparisons.
6. Market-wise price comparisons.
7. Variety and grade analysis.
8. Key findings and limitations.

**Dataset:** `data/processed/onion_dataset_deduplicated_v2.csv`

**Important:** This notebook performs exploratory analysis. It does not modify the original dataset or overwrite existing model evaluation charts."""),
    nbformat.v4.new_markdown_cell("""## 1. Import Libraries

We use pandas for data analysis, NumPy for numerical operations, and Matplotlib for visualizations."""),
    nbformat.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/onion_dataset_deduplicated_v2.csv")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Dataset exists:", DATA_FILE.exists())
print("Figure directory:", FIGURE_DIR)"""),
    nbformat.v4.new_markdown_cell("""## 2. Load the Final Onion Dataset

The final dataset contains the onion records after preprocessing and exact duplicate removal.

We load this dataset for analysis. The original CSV file is not modified."""),
    nbformat.v4.new_code_cell("""if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required_columns = [
    "State", "District", "Market", "Commodity",
    "Variety", "Grade", "Arrival_Date",
    "Min_Price", "Max_Price", "Modal_Price"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"], errors="coerce"
)

for column in ["Min_Price", "Max_Price", "Modal_Price"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Dataset loaded successfully.")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("Shape:", df.shape)

display(df.head())"""),
    nbformat.v4.new_markdown_cell("""## 3. Dataset Structure

Inspect the column names, data types, and dataset size."""),
    nbformat.v4.new_code_cell("""print("COLUMN NAMES")
print(df.columns.tolist())

print("\\nDATA TYPES")
display(df.dtypes.to_frame("Data Type"))

print("\\nDATASET INFORMATION")
print("Records:", f"{df.shape[0]:,}")
print("Features:", df.shape[1])
print("Memory usage (MB):", round(df.memory_usage(deep=True).sum() / (1024**2), 2))"""),
    nbformat.v4.new_markdown_cell("""## 4. Descriptive Statistics

The following statistics describe the onion modal prices.

- Mean: arithmetic average.
- Median: middle observation.
- Standard deviation: variation in prices.
- Minimum and maximum: observed price limits.
- Quartiles: distribution of the observations."""),
    nbformat.v4.new_code_cell("""print("PRICE SUMMARY")
display(df[["Min_Price", "Max_Price", "Modal_Price"]].describe().round(2))

print("\\nUNIQUE VALUES")
for column in ["State", "District", "Market", "Variety", "Grade"]:
    print(f"{column}: {df[column].nunique(dropna=True):,}")"""),
    nbformat.v4.new_markdown_cell("""## 5. Data Quality Checks

Check missing values, exact duplicate rows, invalid dates, and invalid modal prices.

The dataset was already deduplicated in the preprocessing stage. This cell independently verifies that result."""),
    nbformat.v4.new_code_cell("""quality = pd.DataFrame({
    "Missing Values": df.isna().sum(),
    "Missing Percentage": (df.isna().mean() * 100).round(3)
})

display(quality)

print("Exact duplicate rows:", f"{df.duplicated().sum():,}")
print("Invalid arrival dates:", f"{df['Arrival_Date'].isna().sum():,}")
print("Missing modal prices:", f"{df['Modal_Price'].isna().sum():,}")
print("Zero or negative modal prices:",
      f"{(df['Modal_Price'] <= 0).sum():,}")"""),
    nbformat.v4.new_markdown_cell("""## 6. Onion Modal Price Distribution

A histogram helps us understand how frequently different modal price levels occur.

For readability, the chart is limited to the 99th percentile. This only limits the displayed chart range; it does not remove observations from the dataset."""),
    nbformat.v4.new_code_cell("""prices = df["Modal_Price"].dropna()
upper_limit = prices.quantile(0.99)

plt.figure(figsize=(10, 6))
plt.hist(prices[prices <= upper_limit], bins=50)
plt.title("Distribution of Onion Modal Prices")
plt.xlabel("Modal Price (₹ per quintal)")
plt.ylabel("Number of Records")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = FIGURE_DIR / "onion_eda_price_distribution.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print("Chart saved:", output)
print("99th percentile:", round(upper_limit, 2))"""),
    nbformat.v4.new_markdown_cell("""## 7. Monthly Onion Price Trend

Calculate the average modal price and number of records for each month.

The monthly average is calculated across all available onion records. It is not a fixed-market price series, so changes can also reflect changes in the mix of reporting markets."""),
    nbformat.v4.new_code_cell("""eda_df = df.dropna(
    subset=["Arrival_Date", "Modal_Price"]
).copy()

eda_df["Year_Month"] = (
    eda_df["Arrival_Date"].dt.to_period("M").astype(str)
)

monthly_summary = (
    eda_df.groupby("Year_Month")["Modal_Price"]
    .agg(Average_Price="mean", Median_Price="median", Records="count")
    .reset_index()
)

display(monthly_summary.head(12))

plt.figure(figsize=(13, 6))
plt.plot(
    monthly_summary["Year_Month"],
    monthly_summary["Average_Price"],
    marker="o"
)
plt.title("Monthly Average Onion Modal Price")
plt.xlabel("Year and Month")
plt.ylabel("Average Modal Price (₹ per quintal)")
plt.xticks(rotation=60, ha="right")
plt.grid(alpha=0.3)
plt.tight_layout()

output = FIGURE_DIR / "onion_eda_monthly_price_trend.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print("Chart saved:", output)"""),
    nbformat.v4.new_markdown_cell("""## 8. Year-wise Onion Price Summary

Compare the number of observations and average modal price for each year.

Note that 2026 is a partial year in this dataset, ending on April 20, 2026. Its annual average should not be interpreted as a full-year result."""),
    nbformat.v4.new_code_cell("""yearly_summary = (
    eda_df.groupby(eda_df["Arrival_Date"].dt.year)["Modal_Price"]
    .agg(Average_Price="mean", Median_Price="median", Records="count")
    .reset_index()
    .rename(columns={"Arrival_Date": "Year"})
)

display(yearly_summary.round(2))"""),
    nbformat.v4.new_markdown_cell("""## 9. State-wise Price Analysis

Compare states using their average modal price and number of records.

A minimum observation count is used to avoid highlighting states represented by only a small number of records."""),
    nbformat.v4.new_code_cell("""state_summary = (
    eda_df.groupby("State")["Modal_Price"]
    .agg(Average_Price="mean", Median_Price="median", Records="count")
    .reset_index()
)

eligible_states = state_summary[
    state_summary["Records"] >= 100
].copy()

top_states = eligible_states.sort_values(
    "Average_Price", ascending=False
).head(15)

display(top_states.round(2))

if not top_states.empty:
    plot_data = top_states.sort_values("Average_Price")

    plt.figure(figsize=(11, 7))
    plt.barh(plot_data["State"], plot_data["Average_Price"])
    plt.title("15 States with Highest Average Onion Modal Prices")
    plt.xlabel("Average Modal Price (₹ per quintal)")
    plt.ylabel("State")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "onion_eda_state_comparison.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Chart saved:", output)
else:
    print("No states meet the minimum record threshold.")"""),
    nbformat.v4.new_markdown_cell("""## 10. Market-wise Price Analysis

Markets can have different price levels due to local supply, demand, transport, and seasonal conditions.

To make the comparison more reliable, only markets with at least 100 observations are included in the ranking."""),
    nbformat.v4.new_code_cell("""market_summary = (
    eda_df.groupby(["State", "District", "Market"])["Modal_Price"]
    .agg(Average_Price="mean", Median_Price="median", Records="count")
    .reset_index()
)

eligible_markets = market_summary[
    market_summary["Records"] >= 100
].copy()

top_markets = eligible_markets.sort_values(
    "Average_Price", ascending=False
).head(15)

display(top_markets.round(2))"""),
    nbformat.v4.new_markdown_cell("""## 11. Variety and Grade Analysis

Examine how many records are available for each onion variety and grade, and compare their average modal prices.

These are descriptive comparisons and do not establish that variety or grade alone causes a price difference."""),
    nbformat.v4.new_code_cell("""variety_summary = (
    eda_df.groupby("Variety")["Modal_Price"]
    .agg(Average_Price="mean", Records="count")
    .reset_index()
    .sort_values("Records", ascending=False)
)

grade_summary = (
    eda_df.groupby("Grade")["Modal_Price"]
    .agg(Average_Price="mean", Records="count")
    .reset_index()
    .sort_values("Records", ascending=False)
)

print("VARIETY SUMMARY")
display(variety_summary.head(15).round(2))

print("GRADE SUMMARY")
display(grade_summary.round(2))"""),
    nbformat.v4.new_markdown_cell("""## 12. Price Range Analysis

Compare the minimum, maximum, and modal prices.

The average price range is calculated as maximum minus minimum. These fields describe the reported daily market price range, while modal price remains the prediction target in the modeling workflow."""),
    nbformat.v4.new_code_cell("""range_df = eda_df.dropna(
    subset=["Min_Price", "Max_Price", "Modal_Price"]
).copy()

range_df["Price_Range"] = (
    range_df["Max_Price"] - range_df["Min_Price"]
)

print("Average minimum price:", round(range_df["Min_Price"].mean(), 2))
print("Average modal price:", round(range_df["Modal_Price"].mean(), 2))
print("Average maximum price:", round(range_df["Max_Price"].mean(), 2))
print("Average price range:", round(range_df["Price_Range"].mean(), 2))

display(range_df[
    ["Min_Price", "Modal_Price", "Max_Price", "Price_Range"]
].describe().round(2))"""),
    nbformat.v4.new_markdown_cell("""## 13. Key EDA Findings

The following cell generates project findings from the loaded dataset. Interpret these results alongside the charts and record counts."""),
    nbformat.v4.new_code_cell("""print("========== ONION EDA FINDINGS ==========")
print("Total records:", f"{len(df):,}")
print("States:", f"{df['State'].nunique():,}")
print("Markets:", f"{df['Market'].nunique():,}")
print("Districts:", f"{df['District'].nunique():,}")
print("Varieties:", f"{df['Variety'].nunique():,}")

print("\\nModal price mean:", round(prices.mean(), 2))
print("Modal price median:", round(prices.median(), 2))
print("Modal price standard deviation:", round(prices.std(), 2))
print("Modal price 99th percentile:", round(prices.quantile(0.99), 2))

valid_dates = eda_df["Arrival_Date"]
print("\\nDate coverage:", valid_dates.min().date(), "to", valid_dates.max().date())

if not monthly_summary.empty:
    highest_month = monthly_summary.loc[
        monthly_summary["Average_Price"].idxmax()
    ]
    lowest_month = monthly_summary.loc[
        monthly_summary["Average_Price"].idxmin()
    ]

    print("\\nHighest monthly average:")
    print(highest_month.to_string())

    print("\\nLowest monthly average:")
    print(lowest_month.to_string())

print("\\nEDA COMPLETED")"""),
    nbformat.v4.new_markdown_cell("""## 14. EDA Conclusion and Limitations

### Conclusion
This notebook explores the distribution and geographic and temporal variation of onion modal prices using the final deduplicated dataset.

### Important limitations
1. The dataset covers January 1, 2025 to April 20, 2026; it is not a complete long-term historical record.
2. The 2026 period is partial.
3. Monthly and state-wide averages combine observations from different markets. Changes in market composition can influence these averages.
4. A market with more records can have a more stable average than a market with few records.
5. Descriptive patterns do not prove causation.
6. The dataset contains historical observations; the analysis does not provide live market prices.

### Generated EDA charts
- `onion_eda_price_distribution.png`
- `onion_eda_monthly_price_trend.png`
- `onion_eda_state_comparison.png`

### Next notebook
`04_Feature_Engineering.ipynb` – prepare time-based and historical price features for machine learning.""")
]

nb.cells = cells
nbformat.validate(nb)
nbformat.write(nb, OUTPUT)

print("NOTEBOOK CREATED SUCCESSFULLY")
print("File:", OUTPUT)
print("Cells:", len(nb.cells))
print("Markdown cells:", sum(c.cell_type == "markdown" for c in nb.cells))
print("Code cells:", sum(c.cell_type == "code" for c in nb.cells))
print("Size:", f"{OUTPUT.stat().st_size:,}", "bytes")
