import pandas as pd

csv_path = "/mnt/h/Meu Drive/UFRGS/TCC/datalake/processed/Muse-v1.0/Muse-v1.0_filtered.csv"
df = pd.read_csv(csv_path)

print(df.head())
