from src.data.spotify_client import get_spotify_client, get_spotify_public_client
from src.data.data_loader import get_all_liked_songs
from src.features.song_features import get_audio_features
import pandas as pd

sp_user = get_spotify_client()
sp_public = get_spotify_public_client()

print("Descargando canciones...")
songs = get_all_liked_songs(sp_user)

track_ids = [s["id"] for s in songs]

print("Descargando features...")
features = get_audio_features(sp_public, track_ids)

df_songs = pd.DataFrame(songs)
df_features = pd.DataFrame(features)

if df_features.empty:
    raise ValueError("No se han podido obtener audio features")

if "id" not in df_features.columns:
    raise ValueError("La columna 'id' no existe en features")

df = df_songs.merge(df_features, on="id", how="inner")

df.to_csv("data/processed/songs_dataset.csv", index=False)

print("Dataset creado correctamente")