"""
src/model/recommender.py  — v4 (normalización z-score)

Sistema híbrido de recomendación musical con tres capas:

  1. Similitud semántica  — embedding Sentence-BERT enriquecido con tags
     de Last.fm y descriptores textuales de audio features.

  2. Similitud acústica   — solo para canciones con features de Kaggle.
     Compara perfil numérico inferido de la vibe con features reales.

  3. Boost por tags       — bonus para canciones Last.fm cuyo género
     coincide con keywords de la vibe.

Problema corregido en v4:
    Las similitudes semántica y acústica tienen distribuciones muy distintas
    (acústica sistemáticamente más alta: 0.78-0.93 vs semántica 0.30-0.60).
    Sin normalizar, el componente acústico domina el ranking aunque su peso
    declarado sea menor. Se aplica z-score a cada componente antes de
    ponderar, de forma que los pesos reflejen la intención real.

Score = (w_sem * zscore(sim_sem) + w_ac * zscore(sim_ac) + w_tag * zscore(boost)) / sum(w)
        convertido de vuelta a [0,1] para interpretabilidad.
"""

import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from src.features.embedding_model import get_embedding

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "liveness", "loudness",
]

VIBE_PROFILES = {
    "gym":       {"energy": 0.9, "tempo": 0.85, "valence": 0.65, "danceability": 0.7},
    "workout":   {"energy": 0.9, "tempo": 0.85, "valence": 0.6,  "danceability": 0.65},
    "motivat":   {"energy": 0.85,"tempo": 0.8,  "valence": 0.75},
    "entrena":   {"energy": 0.9, "tempo": 0.85, "valence": 0.65},
    "chill":     {"energy": 0.25,"acousticness": 0.7, "valence": 0.55, "danceability": 0.35},
    "study":     {"energy": 0.2, "instrumentalness": 0.6, "acousticness": 0.6, "speechiness": 0.05},
    "estudia":   {"energy": 0.2, "instrumentalness": 0.6, "acousticness": 0.6},
    "relax":     {"energy": 0.2, "acousticness": 0.65, "valence": 0.5},
    "tranquil":  {"energy": 0.2, "acousticness": 0.65, "valence": 0.45},
    "sleep":     {"energy": 0.1, "acousticness": 0.8,  "instrumentalness": 0.7},
    "dormir":    {"energy": 0.1, "acousticness": 0.8,  "instrumentalness": 0.7},
    "party":     {"danceability": 0.9, "energy": 0.85, "valence": 0.85, "tempo": 0.75},
    "fiesta":    {"danceability": 0.9, "energy": 0.85, "valence": 0.85},
    "dance":     {"danceability": 0.9, "energy": 0.8,  "valence": 0.8},
    "bailar":    {"danceability": 0.9, "energy": 0.8,  "valence": 0.8},
    "sad":       {"valence": 0.15, "energy": 0.3, "acousticness": 0.6},
    "triste":    {"valence": 0.15, "energy": 0.3, "acousticness": 0.6},
    "nostalgi":  {"valence": 0.3,  "energy": 0.3, "acousticness": 0.55},
    "night":     {"energy": 0.4,  "valence": 0.4, "acousticness": 0.5},
    "noche":     {"energy": 0.35, "valence": 0.35,"acousticness": 0.55},
    "melancol":  {"valence": 0.15,"energy": 0.25, "acousticness": 0.65},
    "focus":     {"energy": 0.3,  "instrumentalness": 0.65, "speechiness": 0.05},
    "concentra": {"energy": 0.3,  "instrumentalness": 0.65, "speechiness": 0.05},
    "work":      {"energy": 0.35, "instrumentalness": 0.5,  "valence": 0.5},
    "driv":      {"energy": 0.65, "valence": 0.6, "danceability": 0.55, "tempo": 0.6},
    "conducir":  {"energy": 0.65, "valence": 0.6, "danceability": 0.55},
    "road trip": {"energy": 0.7,  "valence": 0.7, "danceability": 0.6},
    "happy":     {"valence": 0.9, "energy": 0.7,  "danceability": 0.75},
    "alegre":    {"valence": 0.9, "energy": 0.7,  "danceability": 0.75},
    "romantic":  {"valence": 0.6, "energy": 0.3,  "acousticness": 0.6, "danceability": 0.45},
    "romanc":    {"valence": 0.6, "energy": 0.3,  "acousticness": 0.6},
    "acoustic":  {"acousticness": 0.9, "energy": 0.25, "instrumentalness": 0.5},
    "acustic":   {"acousticness": 0.9, "energy": 0.25},
    "rap":       {"speechiness": 0.7, "energy": 0.75, "danceability": 0.75},
    "reggaeton": {"danceability": 0.85,"energy": 0.8,  "speechiness": 0.15, "tempo": 0.7},
    "reggaetón": {"danceability": 0.85,"energy": 0.8,  "speechiness": 0.15},
    "pop":       {"danceability": 0.65,"energy": 0.6,  "valence": 0.6},
    "rock":      {"energy": 0.8,  "loudness": 0.75, "acousticness": 0.1},
    "indie":     {"energy": 0.5,  "acousticness": 0.45,"valence": 0.5},
    "electro":   {"energy": 0.8,  "danceability": 0.8, "acousticness": 0.05},
    "piano":     {"acousticness": 0.85,"instrumentalness": 0.7, "energy": 0.2},
}

TAG_KEYWORD_MAP = {
    "pop":               ["pop", "party", "fiesta", "happy", "alegre"],
    "indie pop":         ["indie", "chill", "relax"],
    "indie":             ["indie", "chill", "relax", "night", "noche"],
    "rock":              ["rock", "gym", "workout"],
    "electropop":        ["electro", "dance", "party", "fiesta", "gym"],
    "electronic":        ["electro", "dance", "gym", "workout"],
    "dance":             ["dance", "bailar", "party", "fiesta", "gym"],
    "reggaeton":         ["reggaeton", "reggaetón", "party", "fiesta", "bailar"],
    "latin":             ["reggaeton", "party", "fiesta", "bailar"],
    "hip-hop":           ["rap", "gym", "workout"],
    "rap":               ["rap", "gym", "workout"],
    "r&b":               ["romanc", "night", "noche", "chill"],
    "soul":              ["romanc", "chill", "sad", "triste"],
    "acoustic":          ["acoustic", "acustic", "chill", "relax", "sad"],
    "folk":              ["acoustic", "chill", "relax", "nostalgi"],
    "singer-songwriter": ["acoustic", "sad", "relax", "nostalgi"],
    "sad":               ["sad", "triste", "melancol", "nostalgi"],
    "melancholy":        ["melancol", "sad", "triste"],
    "romantic":          ["romanc", "night", "chill"],
    "classical":         ["focus", "concentra", "study", "piano"],
    "instrumental":      ["focus", "concentra", "study", "sleep"],
    "jazz":              ["chill", "relax", "night", "focus"],
    "ambient":           ["sleep", "dormir", "focus", "relax"],
    "chill":             ["chill", "relax", "study", "sleep"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_embedding(x) -> np.ndarray:
    """Parser robusto para todos los formatos de embedding en CSV."""
    if isinstance(x, np.ndarray):
        return x
    s = str(x)
    s = re.sub(r'np\.float\d+\(([^)]+)\)', r'\1', s)
    s = s.strip().strip('[]').replace(',', ' ')
    return np.fromstring(s, sep=' ')


def _zscore(arr: np.ndarray) -> np.ndarray:
    """
    Z-score sobre el array completo (población = todas las canciones).
    Permite comparar componentes con distribuciones muy distintas.
    Devuelve array con media≈0 y std≈1.
    """
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _zscore_to_01(arr: np.ndarray) -> np.ndarray:
    """Escala z-scores a [0,1] para interpretabilidad del score final."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.full_like(arr, 0.5)
    return (arr - arr_min) / (arr_max - arr_min)


def _infer_acoustic_profile(vibe: str) -> dict | None:
    vibe_lower = vibe.lower()
    for keyword, profile in VIBE_PROFILES.items():
        if keyword in vibe_lower:
            return profile
    return None


def _compute_tag_boost(tags_str: str, vibe: str) -> float:
    if not tags_str or not isinstance(tags_str, str) or tags_str == "nan":
        return 0.0
    vibe_lower = vibe.lower()
    tags = [t.strip().lower() for t in tags_str.split(",")]
    boost = 0.0
    for tag in tags:
        for kw in TAG_KEYWORD_MAP.get(tag, []):
            if kw in vibe_lower:
                boost += 0.1
                break
    return min(boost, 1.0)


def _normalize_features(df: pd.DataFrame) -> tuple:
    available = [
        c for c in AUDIO_FEATURE_COLS
        if c in df.columns and df[c].notna().any()
    ]
    if not available:
        return None, []
    matrix = df[available].copy().fillna(df[available].median())
    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix), available


# ── Carga ──────────────────────────────────────────────────────────────────────
def load_dataset(path: str = "data/processed/songs_dataset_enriched.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(_parse_embedding)
    for col in AUDIO_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lastfm_tags" not in df.columns:
        df["lastfm_tags"] = ""
    if "audio_source" not in df.columns:
        df["audio_source"] = "none"
    return df


# ── Recomendación ──────────────────────────────────────────────────────────────
def recommend_songs(
    vibe: str,
    df: pd.DataFrame,
    top_n: int = 10,
    max_per_artist: int = 2,
    weight_semantic: float = 0.65,
    weight_acoustic: float = 0.25,
    weight_tags: float = 0.10,
) -> pd.DataFrame:
    """
    Genera recomendaciones híbridas con scores normalizados por z-score,
    de forma que cada componente contribuya en la proporción real indicada
    por sus pesos, independientemente de su escala absoluta.

    Parámetros
    ----------
    vibe            : Descripción libre del usuario
    df              : DataFrame cargado con load_dataset()
    top_n           : Número de recomendaciones finales
    max_per_artist  : Máximo canciones por artista (diversidad)
    weight_semantic : Peso real de la similitud semántica (default 0.65)
    weight_acoustic : Peso real de la similitud acústica  (default 0.25)
    weight_tags     : Peso real del boost por tags        (default 0.10)
    """

    # ── 1. Similitud semántica ─────────────────────────────────────────────────
    vibe_embedding = get_embedding(vibe)
    song_embeddings = np.vstack(df["embedding"].values)
    semantic_sim = cosine_similarity([vibe_embedding], song_embeddings)[0]

    # ── 2. Similitud acústica (solo canciones con features de Kaggle) ──────────
    feature_matrix, available_features = _normalize_features(df)
    acoustic_sim = np.zeros(len(df))
    w_acoustic = weight_acoustic

    if feature_matrix is not None:
        profile = _infer_acoustic_profile(vibe)
        if profile:
            vibe_vector = np.full(len(available_features), 0.5)
            for i, feat in enumerate(available_features):
                if feat in profile:
                    vibe_vector[i] = profile[feat]
            raw_acoustic = cosine_similarity([vibe_vector], feature_matrix)[0]
            has_kaggle = df["audio_source"].eq("kaggle").values
            acoustic_sim = raw_acoustic * has_kaggle
        else:
            w_acoustic = 0.0

    # ── 3. Boost por tags Last.fm ──────────────────────────────────────────────
    tag_boost = np.array([
        _compute_tag_boost(str(row.get("lastfm_tags", "")), vibe)
        for _, row in df.iterrows()
    ])

    # ── 4. Normalización z-score + score final ─────────────────────────────────
    # Cada componente se normaliza independientemente (media=0, std=1)
    # para que los pesos reflejen su contribución real al ranking.
    sem_z  = _zscore(semantic_sim)
    ac_z   = _zscore(acoustic_sim) if w_acoustic > 0 else np.zeros(len(df))
    tag_z  = _zscore(tag_boost)    if tag_boost.std() > 1e-8 else tag_boost

    total_w = weight_semantic + w_acoustic + weight_tags
    score_z = (
        weight_semantic * sem_z  +
        w_acoustic      * ac_z   +
        weight_tags     * tag_z
    ) / total_w

    # Convertir a [0,1] para interpretabilidad
    score_01 = _zscore_to_01(score_z)

    # ── 5. Construir dataframe con scores ──────────────────────────────────────
    df = df.copy()
    df["similarity"]          = score_01
    df["similarity_semantic"] = _zscore_to_01(sem_z)
    df["similarity_acoustic"] = _zscore_to_01(ac_z) if w_acoustic > 0 else acoustic_sim
    df["tag_boost"]           = tag_boost

    df_sorted = df.sort_values(by="similarity", ascending=False)

    # ── 6. Diversidad por artista ──────────────────────────────────────────────
    recommendations = []
    artist_count = {}
    for _, row in df_sorted.iterrows():
        artist = row["artist"]
        if artist_count.get(artist, 0) < max_per_artist:
            recommendations.append(row)
            artist_count[artist] = artist_count.get(artist, 0) + 1
        if len(recommendations) >= top_n:
            break

    result_cols = ["name", "artist", "similarity", "audio_source",
                   "similarity_semantic", "similarity_acoustic", "tag_boost"]
    for col in ["energy", "valence", "danceability", "tempo"]:
        if col in df.columns:
            result_cols.append(col)

    return pd.DataFrame(recommendations)[result_cols]