import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.features.embedding_model import get_embedding


def load_dataset(path="data/processed/songs_dataset.csv"):
    df = pd.read_csv(path)
    
    # convertir embeddings de string a array
    df["embedding"] = df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    
    return df


"""def recommend_songs(vibe, df, top_n=20):
    # embedding del usuario
    vibe_embedding = get_embedding(vibe)
    
    # matriz de embeddings
    song_embeddings = np.vstack(df["embedding"].values)
    
    # calcular similitud
    similarities = cosine_similarity([vibe_embedding], song_embeddings)[0]
    
    # añadir al dataframe
    df["similarity"] = similarities
    
    # ordenar
    recommendations = df.sort_values(by="similarity", ascending=False)
    
    return recommendations[["name", "artist", "similarity"]].head(top_n)"""

def recommend_songs(vibe, df, top_n=10, max_per_artist=2):
    vibe_embedding = get_embedding(vibe)
    
    song_embeddings = np.vstack(df["embedding"].values)
    similarities = cosine_similarity([vibe_embedding], song_embeddings)[0]
    
    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False)
    
    recommendations = []
    artist_count = {}
    
    for _, row in df_sorted.iterrows():
        artist = row["artist"]
        
        if artist_count.get(artist, 0) < max_per_artist:
            recommendations.append(row)
            artist_count[artist] = artist_count.get(artist, 0) + 1
        
        if len(recommendations) >= top_n:
            break
    
    return pd.DataFrame(recommendations)[["name", "artist", "similarity"]]