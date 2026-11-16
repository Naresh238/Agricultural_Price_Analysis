import pandas as pd

files = [
    "reports/onion_hybrid_test_comparison.csv",
    "reports/onion_hybrid_test_predictions.csv",
    "reports/onion_hybrid_market_improvement_analysis.csv"
]

for file in files:
    print("\n" + "=" * 70)
    print("FILE:", file)
    df = pd.read_csv(file, nrows=5)
    print("COLUMNS:", df.columns.tolist())
    print(df.head(2).to_string(index=False))
