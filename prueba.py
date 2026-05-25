import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/songs_dataset_enriched.csv')
sample = df['embedding'].iloc[0]
print('Tipo:', type(sample))
print('Valor (primeros 80 chars):', str(sample)[:80])