import pandas as pd

# Load dataset dari CSV
df = pd.read_csv(r"dataset_final.csv")

# Simpan dataset ke file pickle (.pkl)
df.to_pickle(r"dataset_final.pkl")

print("Konversi CSV ke PKL selesai!")
