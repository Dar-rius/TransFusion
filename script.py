import pandas as pd

dataset = pd.read_csv("./data/metric_market.csv")
regime_mapping = {
        "Stable": 0.0,
        "Volatility": 1.0,
        "Crisis": 2.0
        }

#dataset["state"] = dataset["state"].map(regime_mapping)
#dataset["state"] = dataset["state"].astype(float)
dataset = dataset.drop("Unnamed: 0.1", axis=1)
dataset.to_csv("./data/metric_market.csv")
