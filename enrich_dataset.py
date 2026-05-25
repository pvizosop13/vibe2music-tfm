"""
enrich_dataset.py

Enriquece songs_dataset.csv con dos fuentes complementarias:

  1. KAGGLE LOOKUP  — join por (name, artist) normalizado contra
     data/external/spotify_kaggle.csv  (~114k canciones con audio features).
     Cubre canciones mainstream/internacionales.

  2. LAST.FM TAGS   — para las canciones sin match en Kaggle, consulta la
     API de Last.fm y obtiene los top tags del artista (genre proxies).
     Cubre artistas hispanófonos/nicho que no están en Kaggle.

Resultado: data/processed/songs_dataset_enriched.csv con columnas nuevas:
    danceability, energy, valence, tempo, acousticness,
    instrumentalness, speechiness, liveness, loudness,
    lastfm_tags, audio_source  ("kaggle" | "lastfm_tags" | "none")

Uso:
    python enrich_dataset.py
"""

import os
import time
import re
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
KAGGLE_PATH    = "data/external/spotify_kaggle.csv"
INPUT_PATH     = "data/processed/songs_dataset.csv"
OUTPUT_PATH    = "data/processed/songs_dataset_enriched.csv"

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE    = "https://ws.audioscrobbler.com/2.0/"

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "liveness", "loudness",
]

LASTFM_DELAY = 0.25   # segundos entre llamadas a Last.fm


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize_str(s: str) -> str:
    """Minúsculas, sin tildes básicas, sin paréntesis ni feat., strip."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    # quitar secciones tipo "(feat. X)" o "(remix)" del título
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"\[.*?\]", "", s).strip()
    # tildes comunes
    for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
        s = s.replace(a, b)
    # quitar puntuación extra
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_kaggle(path: str) -> pd.DataFrame:
    print(f"Cargando dataset de Kaggle desde {path}...")
    df = pd.read_csv(path)

    # Normalizar nombres de columnas (el CSV de maharshipandya usa 'artists' y 'track_name')
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("artists", "artist_name", "artist"):
            col_map[c] = "artist"
        elif cl in ("track_name", "name", "song_name", "title"):
            col_map[c] = "name"
        elif cl in ("track_id", "id"):
            col_map[c] = "track_id"
    df = df.rename(columns=col_map)

    # Crear clave de join normalizada
    df["_key"] = df.apply(
        lambda r: normalize_str(str(r.get("name", ""))) + "|||" +
                  normalize_str(str(r.get("artist", "")).split(";")[0].split(",")[0]),
        axis=1
    )

    # Asegurar que las columnas de features son numéricas
    for col in AUDIO_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  {len(df)} canciones en el dataset de Kaggle.")
    return df


def get_lastfm_tags(artist: str, track: str = None) -> list[str]:
    """
    Obtiene top tags de Last.fm. Intenta primero por track, luego por artista.
    Devuelve lista de tags en minúsculas (máx. 5).
    """
    if not LASTFM_API_KEY:
        return []

    # Intentar con track primero si tenemos nombre
    if track:
        params = {
            "method": "track.getTopTags",
            "artist": artist,
            "track": track,
            "api_key": LASTFM_API_KEY,
            "format": "json",
            "limit": 5,
        }
        try:
            r = requests.get(LASTFM_BASE, params=params, timeout=8)
            if r.status_code == 200:
                data = r.json()
                tags = data.get("toptags", {}).get("tag", [])
                if tags:
                    return [t["name"].lower() for t in tags[:5]]
        except Exception:
            pass

    # Fallback: tags por artista
    params = {
        "method": "artist.getTopTags",
        "artist": artist,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": 5,
    }
    try:
        r = requests.get(LASTFM_BASE, params=params, timeout=8)
        if r.status_code == 200:
            data = r.json()
            tags = data.get("toptags", {}).get("tag", [])
            return [t["name"].lower() for t in tags[:5]]
    except Exception:
        pass

    return []


# ── Pipeline principal ─────────────────────────────────────────────────────────
def enrich_dataset():
    # 1. Cargar dataset personal
    print(f"Cargando dataset personal desde {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    print(f"  {len(df)} canciones.")

    # Inicializar columnas nuevas
    for col in AUDIO_FEATURE_COLS:
        df[col] = np.nan
    df["lastfm_tags"] = ""
    df["audio_source"] = "none"

    # 2. Cargar y cruzar con Kaggle
    kaggle_df = load_kaggle(KAGGLE_PATH)
    kaggle_lookup = kaggle_df.drop_duplicates(subset=["_key"]).set_index("_key")

    df["_key"] = df.apply(
        lambda r: normalize_str(str(r.get("name", ""))) + "|||" +
                  normalize_str(str(r.get("artist", ""))),
        axis=1
    )

    kaggle_hits = 0
    for idx, row in df.iterrows():
        key = row["_key"]
        if key in kaggle_lookup.index:
            match = kaggle_lookup.loc[key]
            for col in AUDIO_FEATURE_COLS:
                if col in kaggle_lookup.columns:
                    df.at[idx, col] = match[col]
            df.at[idx, "audio_source"] = "kaggle"
            kaggle_hits += 1

    print(f"\nKaggle lookup: {kaggle_hits}/{len(df)} canciones con match "
          f"({100*kaggle_hits/len(df):.1f}%)")

    # 3. Last.fm para las que no tienen match
    no_match = df[df["audio_source"] == "none"]
    print(f"\nConsultando Last.fm para {len(no_match)} canciones sin features...")

    if not LASTFM_API_KEY:
        print("  [WARN] LASTFM_API_KEY no encontrada en .env — saltando Last.fm")
    else:
        lastfm_hits = 0
        # Cachear por artista para no repetir llamadas
        artist_tags_cache = {}

        for i, (idx, row) in enumerate(no_match.iterrows()):
            artist = str(row.get("artist", ""))
            name   = str(row.get("name", ""))

            if artist not in artist_tags_cache:
                tags = get_lastfm_tags(artist, name)
                artist_tags_cache[artist] = tags
                time.sleep(LASTFM_DELAY)
            else:
                tags = artist_tags_cache[artist]

            if tags:
                df.at[idx, "lastfm_tags"] = ", ".join(tags)
                df.at[idx, "audio_source"] = "lastfm_tags"
                lastfm_hits += 1

            if (i + 1) % 50 == 0:
                print(f"  Last.fm: {i+1}/{len(no_match)} | con tags: {lastfm_hits}")

        print(f"  Last.fm: {lastfm_hits}/{len(no_match)} artistas con tags")

    # 4. Resumen final
    source_counts = df["audio_source"].value_counts()
    total = len(df)
    print("\n── Resumen de cobertura ──────────────────────────────")
    for src, count in source_counts.items():
        print(f"  {src:15s}: {count:5d} canciones ({100*count/total:.1f}%)")

    # 5. Guardar
    # Convertir embeddings de vuelta a string para CSV
    df["embedding"] = df["embedding"].apply(lambda x: str(list(x)))
    df = df.drop(columns=["_key"], errors="ignore")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset enriquecido guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    enrich_dataset()