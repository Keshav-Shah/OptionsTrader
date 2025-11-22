import pickle
from granger_causality import run_granger_on_dict

print("Loading price data...")
with open("stock_data_dict.pkl", "rb") as f:
    data = pickle.load(f)

print("Running FAST Granger causality analysis (lag=1)...")
df = run_granger_on_dict(data, max_lag=1)   # max_lag ignored internally

print("Saving results...")
df.to_pickle("granger_results.pkl")
df.to_csv("granger_results.csv", index=False)

print("Done!")
print(df.head())
