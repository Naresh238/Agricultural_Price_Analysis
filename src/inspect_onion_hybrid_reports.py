import pandas as pd
from pathlib import Path

files = {
    "comparison": "reports/onion_hybrid_test_comparison.csv",
    "predictions": "reports/onion_hybrid_test_predictions.csv",
    "markets": "reports/onion_hybrid_market_test_comparison.csv",
    "selection": "reports/onion_hybrid_validation_selection.csv"
}

for name, filename in files.items():
    path = Path(filename)
    df = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))
