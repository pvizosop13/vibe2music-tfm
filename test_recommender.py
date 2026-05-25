"""
test_recommender.py  (versión actualizada)

Prueba el sistema híbrido con varias vibes y muestra:
  - Cobertura de fuentes (kaggle / lastfm_tags / none)
  - Resultados de recomendación con scores desglosados
  - Ejemplos con vibes en español e inglés
"""

from src.model.recommender import load_dataset, recommend_songs
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# Cargar dataset enriquecido
print("Cargando dataset...")
df = load_dataset()

# Resumen de cobertura
print(f"\nTotal canciones: {len(df)}")
if "audio_source" in df.columns:
    print("Fuentes de datos:")
    print(df["audio_source"].value_counts().to_string())

# Test con varias vibes
vibes = [
    "chill music for studying",
    "gym motivation high energy",
    "fiesta reggaeton bailar",
    "música tranquila para conducir de noche",
    "sad songs heartbreak",
    "pop español alegre",
]

for vibe in vibes:
    print(f"\n{'='*60}")
    print(f"VIBE: '{vibe}'")
    print('='*60)
    results = recommend_songs(vibe, df, top_n=5)
    print(results[["name", "artist", "similarity", "audio_source",
                   "similarity_semantic", "similarity_acoustic"]].to_string(index=False))