import pandas as pd

df = pd.read_json("categorise_data.json")

counts = df["category"].value_counts()

print(counts)