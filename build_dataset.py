"""
build_dataset.py

Descarga las canciones de Spotify (liked songs), genera embeddings semánticos
y enriquece cada canción con audio features via ReccoBeats API.

Resultado: data/processed/songs_dataset.csv con columnas:
    id, name, artist, text, embedding,
    danceability, energy, valence, tempo, acousticness,
    instrumentalness, speechiness, liveness, loudness, key, mode, time_signature
"""

from src.data.spotify_client import get_spotify_client
from src.data.data_loader import get_all_liked_songs
from src.features.embedding_model import get_embedding
import pandas as pd
import requests
import time
import os

# ── Configuración ──────────────────────────────────────────────────────────────
RECCOBEATS_BASE = "https://api.reccobeats.com"

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "liveness", "loudness", "key", "mode", "time_signature",
]


# ── Funciones auxiliares ────────────────────────────────────────────────────────
def get_audio_features_reccobeats(track_id: str) -> dict | None:
    """Obtiene audio features de ReccoBeats usando el Spotify track ID."""
    url = f"{RECCOBEATS_BASE}/v1/track/{track_id}/audio-features"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {col: data.get(col) for col in AUDIO_FEATURE_COLS}
        return None
    except requests.exceptions.RequestException:
        return None


# ── Pipeline principal ──────────────────────────────────────────────────────────
sp = get_spotify_client()

print("Descargando canciones de Spotify...")
songs = get_all_liked_songs(sp)
df = pd.DataFrame(songs)
print(f"  {len(df)} canciones descargadas.")

# Deduplicar antes de procesar
df = df.drop_duplicates(subset=["name", "artist"])
print(f"  {len(df)} canciones únicas tras deduplicar.")

# Texto representativo para el embedding semántico
df["text"] = df["name"] + " by " + df["artist"] + " genre music mood"

# Embeddings semánticos
print("\nGenerando embeddings semánticos...")
df["embedding"] = df["text"].apply(get_embedding)
print("  Embeddings generados.")

# Inicializar columnas de audio features
for col in AUDIO_FEATURE_COLS:
    df[col] = None

# Audio features via ReccoBeats
print("\nObteniendo audio features via ReccoBeats...")
success, failed = 0, 0

for idx, row in df.iterrows():
    track_id = row.get("id")
    if not track_id:
        continue

    features = get_audio_features_reccobeats(track_id)
    if features:
        for col, val in features.items():
            df.at[idx, col] = val
        success += 1
    else:
        failed += 1

    processed = success + failed
    if processed % 20 == 0:
        print(f"  {processed}/{len(df)} | con features: {success} | sin datos: {failed}")

    time.sleep(0.3)  # respetar rate limit

print(f"\nAudio features: {success}/{len(df)} canciones con datos ({100*success/len(df):.1f}%)")

# Guardar dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/songs_dataset.csv", index=False)
print("\nDataset guardado en data/processed/songs_dataset.csv")
print(f"Columnas: {list(df.columns)}")