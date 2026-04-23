from src.data.spotify_client import get_spotify_client
from src.data.data_loader import get_all_liked_songs
from src.features.embedding_model import get_embedding
import pandas as pd

sp = get_spotify_client()

print("Descargando canciones...")
songs = get_all_liked_songs(sp)

df = pd.DataFrame(songs)

# Creamos texto representativo
df["text"] = df["name"] + " " + df["artist"]

print("Generando embeddings...")
df["embedding"] = df["text"].apply(get_embedding)

# Guardar dataset
df.to_csv("data/processed/songs_dataset.csv", index=False)

print("Dataset con embeddings creado correctamente")