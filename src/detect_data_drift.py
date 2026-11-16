from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "onion_modeling_dataset_v1.csv"
OUTPUT_DIR = ROOT / "reports" / "drift_detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

if "Dataset_Split" not in df.columns:
    raise ValueError("Dataset_Split column is missing.")

reference = df[df["Dataset_Split"].str.lower() == "train"].copy()
comparison = df[df["Dataset_Split"].str.lower() == "test"].copy()

if reference.empty or comparison.empty:
    raise ValueError("Training or test data is empty.")

# Features used by the multi-market model.
numeric_features = [
    "Lag_1",
    "Lag_2",
    "Rolling_Mean_7",
    "Rolling_Mean_30",
    "Previous_Price_Change",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
]

categorical_features = [
    "State",
    "District",
    "Market",
    "Variety",
    "Grade",
]

target_feature = "Modal_Price"

def calculate_psi(reference_values, comparison_values, bins=10):
    """Calculate PSI using reference-data quantile boundaries."""

    ref = pd.to_numeric(
        pd.Series(reference_values), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    comp = pd.to_numeric(
        pd.Series(comparison_values), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if ref.empty or comp.empty:
        return np.nan

    boundaries = np.unique(
        np.quantile(ref, np.linspace(0, 1, bins + 1))
    )

    if len(boundaries) < 2:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    ref_bins = pd.cut(ref, bins=boundaries, include_lowest=True)
    comp_bins = pd.cut(comp, bins=boundaries, include_lowest=True)

    ref_counts = ref_bins.value_counts(sort=False).to_numpy(dtype=float)
    comp_counts = comp_bins.value_counts(sort=False).to_numpy(dtype=float)

    ref_pct = ref_counts / ref_counts.sum()
    comp_pct = comp_counts / comp_counts.sum()

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    comp_pct = np.clip(comp_pct, epsilon, None)

    return float(
        np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct))
    )


def calculate_categorical_psi(reference_values, comparison_values):
    """Calculate PSI for categorical distributions."""

    ref = (
        pd.Series(reference_values)
        .fillna("__MISSING__")
        .astype(str)
    )

    comp = (
        pd.Series(comparison_values)
        .fillna("__MISSING__")
        .astype(str)
    )

    categories = sorted(set(ref.unique()) | set(comp.unique()))

    ref_pct = (
        ref.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    comp_pct = (
        comp.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    comp_pct = np.clip(comp_pct, epsilon, None)

    return float(
        np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct))
    )


results = []

# Numerical model features.
for feature in numeric_features + [target_feature]:

    if feature not in df.columns:
        print(f"WARNING: Missing numerical feature: {feature}")
        continue

    psi = calculate_psi(
        reference[feature],
        comparison[feature]
    )

    results.append({
        "Feature": feature,
        "Feature_Type": (
            "Target" if feature == target_feature else "Numerical"
        ),
        "PSI": psi,
        "Reference_Records": len(reference),
        "Comparison_Records": len(comparison),
    })

# Categorical model features.
for feature in categorical_features:

    if feature not in df.columns:
        print(f"WARNING: Missing categorical feature: {feature}")
        continue

    psi = calculate_categorical_psi(
        reference[feature],
        comparison[feature]
    )

    results.append({
        "Feature": feature,
        "Feature_Type": "Categorical",
        "PSI": psi,
        "Reference_Records": len(reference),
        "Comparison_Records": len(comparison),
    })

report = pd.DataFrame(results)

def classify_drift(psi):
    if pd.isna(psi):
        return "Not Available"
    if psi < 0.10:
        return "Low"
    if psi < 0.20:
        return "Moderate"
    return "High"

report["Drift_Status"] = report["PSI"].apply(classify_drift)

report = report.sort_values(
    "PSI", ascending=False
).reset_index(drop=True)

output_file = OUTPUT_DIR / "train_vs_test_drift_report.csv"
report.to_csv(output_file, index=False)

print("\nDATA DRIFT ANALYSIS COMPLETED")
print("-----------------------------------")
print("Reference records:", len(reference))
print("Comparison records:", len(comparison))
print("Reference period:",
      reference["Arrival_Date"].min(), "to",
      reference["Arrival_Date"].max())
print("Comparison period:",
      comparison["Arrival_Date"].min(), "to",
      comparison["Arrival_Date"].max())

print("\nDRIFT RESULTS")
print(report.to_string(index=False))

print("\nDRIFT STATUS COUNTS")
print(report["Drift_Status"].value_counts().to_string())

print("\nREPORT SAVED:")
print(output_file)

print("\nInterpretation:")
print("PSI < 0.10: Low distribution change")
print("PSI 0.10-0.20: Moderate distribution change")
print("PSI >= 0.20: High distribution change")
print("These thresholds are screening guidelines, not automatic retraining rules.")
