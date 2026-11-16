import pandas as pd

file = "data/processed/onion_prices.csv"

df = pd.read_csv(file)

df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

print("\n20 LOWEST MODAL PRICES")
print("======================")

print(
    df.nsmallest(20, "Modal_Price")[
        ["State", "District", "Market", "Variety",
         "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]
    ].to_string(index=False)
)

print("\n20 HIGHEST MODAL PRICES")
print("=======================")

print(
    df.nlargest(20, "Modal_Price")[
        ["State", "District", "Market", "Variety",
         "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"]
    ].to_string(index=False)
)

print("\nPRICE RELATIONSHIP CHECK")
print("========================")

print("Modal below minimum:",
      (df["Modal_Price"] < df["Min_Price"]).sum())

print("Modal above maximum:",
      (df["Modal_Price"] > df["Max_Price"]).sum())

print("Zero modal prices:",
      (df["Modal_Price"] == 0).sum())

print("\nINVESTIGATION COMPLETED")
