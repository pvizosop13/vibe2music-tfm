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


def recommend_songs(vibe, df, top_n=10):
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
    
    return recommendations[["name", "artist", "similarity"]].head(top_n)