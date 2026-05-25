"""
rebuild_embeddings.py

Regenera los embeddings del dataset enriquecido usando texto más rico
que incluye los tags de Last.fm y las audio features de Kaggle.

Texto anterior (pobre):
    "Song Name by Artist genre music mood"

Texto nuevo (rico):
    "Song Name by Artist | tags: pop, indie pop, chill | mood: calm relaxed | energy: low | feel: positive"

Esto permite que el embedding semántico capture el género, el mood
y las características musicales, no solo el título de la canción.

Uso:
    python rebuild_embeddings.py
"""

import pandas as pd
import numpy as np
from src.features.embedding_model import get_embedding

INPUT_PATH  = "data/processed/songs_dataset_enriched.csv"
OUTPUT_PATH = "data/processed/songs_dataset_enriched.csv"  # sobreescribe


# ── Mapeo de audio features a descriptores textuales ──────────────────────────
def energy_label(val):
    if pd.isna(val): return None
    if val >= 0.75: return "high energy"
    if val >= 0.45: return "medium energy"
    return "low energy"

def valence_label(val):
    if pd.isna(val): return None
    if val >= 0.70: return "happy upbeat positive"
    if val >= 0.40: return "neutral mood"
    return "sad melancholic negative"

def danceability_label(val):
    if pd.isna(val): return None
    if val >= 0.75: return "very danceable"
    if val >= 0.50: return "danceable"
    return "not danceable"

def tempo_label(val):
    if pd.isna(val): return None
    if val >= 140: return "fast tempo"
    if val >= 100: return "medium tempo"
    return "slow tempo"

def acousticness_label(val):
    if pd.isna(val): return None
    if val >= 0.70: return "acoustic"
    if val <= 0.20: return "electronic"
    return None

def instrumentalness_label(val):
    if pd.isna(val): return None
    if val >= 0.50: return "instrumental"
    return None

def speechiness_label(val):
    if pd.isna(val): return None
    if val >= 0.40: return "rap spoken word"
    return None


def build_rich_text(row) -> str:
    """
    Construye un texto descriptivo rico para cada canción combinando:
    - nombre y artista
    - tags de Last.fm (género, mood)
    - descriptores textuales de audio features
    """
    parts = [f"{row['name']} by {row['artist']}"]

    # Tags de Last.fm
    tags = str(row.get("lastfm_tags", ""))
    if tags and tags != "nan":
        parts.append(f"genre {tags}")

    # Descriptores de audio features (si existen)
    descriptors = []
    for fn, col in [
        (energy_label,           "energy"),
        (valence_label,          "valence"),
        (danceability_label,     "danceability"),
        (tempo_label,            "tempo"),
        (acousticness_label,     "acousticness"),
        (instrumentalness_label, "instrumentalness"),
        (speechiness_label,      "speechiness"),
    ]:
        label = fn(row.get(col))
        if label:
            descriptors.append(label)

    if descriptors:
        parts.append(" ".join(descriptors))

    return " | ".join(parts)


# ── Pipeline ───────────────────────────────────────────────────────────────────
def rebuild():
    print(f"Cargando {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} canciones.")

    # Convertir features a float
    for col in ["energy", "valence", "danceability", "tempo",
                "acousticness", "instrumentalness", "speechiness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mostrar ejemplos del texto nuevo antes de embedear
    print("\nEjemplos de texto enriquecido:")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  → {build_rich_text(row)}")

    print("\nRegenerando embeddings con texto enriquecido...")
    df["text"] = df.apply(build_rich_text, axis=1)
    df["embedding"] = df["text"].apply(get_embedding)

    print("Guardando...")
    # Guardar embedding como string compatible
    df["embedding"] = df["embedding"].apply(lambda x: " ".join(map(str, x)))
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset guardado en {OUTPUT_PATH}")
    print("\nAhora ejecuta: python test_recommender.py")


if __name__ == "__main__":
    rebuild()