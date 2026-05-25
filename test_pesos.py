"""
test_pesos.py  — ejecutar desde la raíz del proyecto

Muestra top 20 con desglose de scores para analizar el balance
semántico vs acústico, y prueba diferentes configuraciones de pesos.
"""
import sys
sys.path.insert(0, ".")

from src.model.recommender import load_dataset, recommend_songs
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 35)

print("Cargando dataset...")
df = load_dataset()

# ── Configuraciones de pesos a comparar ───────────────────────────────────────
configs = [
    ("Actual    (sem=0.55 ac=0.35 tag=0.10)", 0.55, 0.35, 0.10),
    ("Más sem   (sem=0.70 ac=0.20 tag=0.10)", 0.70, 0.20, 0.10),
    ("Solo sem  (sem=1.00 ac=0.00 tag=0.00)", 1.00, 0.00, 0.00),
    ("Balancead (sem=0.60 ac=0.25 tag=0.15)", 0.60, 0.25, 0.15),
]

vibes_test = [
    "gym motivation high energy",
    "chill music for studying",
    "música tranquila para conducir de noche",
]

for vibe in vibes_test:
    print(f"\n{'#'*70}")
    print(f"  VIBE: '{vibe}'")
    print(f"{'#'*70}")

    for label, ws, wa, wt in configs:
        print(f"\n  [{label}]")
        results = recommend_songs(
            vibe, df, top_n=20,
            max_per_artist=2,
            weight_semantic=ws,
            weight_acoustic=wa,
            weight_tags=wt,
        )
        # Mostrar top 10 con desglose
        cols = ["name", "artist", "audio_source", "similarity",
                "similarity_semantic", "similarity_acoustic", "tag_boost"]
        print(results[cols].head(10).to_string(index=False))