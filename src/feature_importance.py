import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

print("Loading trained XGBoost model...")

model = joblib.load("models/xgboost_historical_experiment.pkl")

pipeline = model.regressor_
preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importance_values = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Aggregate one-hot encoded categories into original columns
def original_feature(name):
    name = name.replace("categorical__", "")
    name = name.replace("numeric__", "")

    for column in [
        "State", "District", "Market", "Variety", "Grade"
    ]:
        if name.startswith(column + "_"):
            return column

    return name

importance_df["Original_Feature"] = (
    importance_df["Feature"].apply(original_feature)
)

summary = (
    importance_df.groupby("Original_Feature")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

summary.to_csv(
    "reports/xgboost_feature_importance.csv",
    index=False
)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 7))
plt.barh(
    summary["Original_Feature"].iloc[::-1],
    summary["Importance"].iloc[::-1]
)
plt.xlabel("Total Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Onion Price Prediction")
plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png",
    dpi=300
)

plt.close()

print("\nFEATURE IMPORTANCE")
print("==================")
print(summary.to_string(index=False))

print("\nSaved:")
print("reports/xgboost_feature_importance.csv")
print("reports/figures/feature_importance.png")

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED")
