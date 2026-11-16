import pandas as pd
from pathlib import Path

source = Path("data/processed/onion_prices.csv")
output = Path("data/processed/onion_outlier_review.csv")

df = pd.read_csv(source)

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Identify records requiring investigation
df["Duplicate_Record"] = df.duplicated(keep="first")

df["Modal_Below_Min"] = (
    df["Modal_Price"] < df["Min_Price"]
)

df["Modal_Above_Max"] = (
    df["Modal_Price"] > df["Max_Price"]
)

df["Extreme_Price"] = (
    df["Modal_Price"] > 20000
)

df["Very_Low_Price"] = (
    df["Modal_Price"] < 100
)

review = df[
    df["Duplicate_Record"]
    | df["Modal_Below_Min"]
    | df["Modal_Above_Max"]
    | df["Extreme_Price"]
    | df["Very_Low_Price"]
].copy()

review.to_csv(output, index=False)

print("\nOUTLIER REVIEW FILE CREATED")
print("============================")
print("Original records:", f"{len(df):,}")
print("Records flagged:", f"{len(review):,}")
print("Duplicate records:", df.duplicated().sum())
print("Modal below minimum:", df["Modal_Below_Min"].sum())
print("Modal above maximum:", df["Modal_Above_Max"].sum())
print("Extreme prices:", df["Extreme_Price"].sum())
print("Very low prices:", df["Very_Low_Price"].sum())
print("Output:", output)

print("\nReview file saved successfully.")
