import pandas as pd

file = "data/processed/onion_prices.csv"
df = pd.read_csv(file)

print("\nONION PRICE DISTRIBUTION")
print("========================")

print(df[["Min_Price", "Max_Price", "Modal_Price"]].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).round(2).to_string())

print("\nMODAL PRICE THRESHOLD COUNTS")
print("============================")

for limit in [100, 500, 10000, 20000, 50000, 100000]:
    print(
        f"Prices above Rs.{limit:,}:",
        (df["Modal_Price"] > limit).sum()
    )

print("\nRECORDS WITH MODAL PRICE ABOVE Rs. 20,000")
print("=========================================")

high = df[df["Modal_Price"] > 20000]

print("Total:", len(high))

print(
    high.groupby(["State", "Market", "Variety"])["Modal_Price"]
    .agg(["count", "min", "median", "max"])
    .sort_values("count", ascending=False)
    .head(20)
    .round(2)
    .to_string()
)

print("\nEXACT DUPLICATES:", df.duplicated().sum())

print("\nINVESTIGATION COMPLETED")
