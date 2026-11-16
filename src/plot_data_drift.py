from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "reports" / "drift_detection" / "train_vs_test_drift_report.csv"
OUTPUT_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = pd.read_csv(INPUT_FILE)

# Exclude calendar fields from this focused chart.
calendar_features = ["Month", "DayOfYear"]
chart_data = report[~report["Feature"].isin(calendar_features)].copy()
chart_data = chart_data.sort_values("PSI", ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(chart_data["Feature"], chart_data["PSI"])
ax.axvline(0.10, linestyle="--", label="Low / Moderate threshold (0.10)")
ax.axvline(0.20, linestyle=":", label="Moderate / High threshold (0.20)")

ax.set_title("Train vs Test Data Drift (Excluding Calendar Fields)")
ax.set_xlabel("Population Stability Index (PSI)")
ax.set_ylabel("Feature")
ax.legend()
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()

output_file = OUTPUT_DIR / "onion_data_drift_psi.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print("DATA DRIFT CHART CREATED")
print(output_file)
print("Features plotted:", len(chart_data))
