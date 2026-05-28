import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.model.recommender import load_dataset, recommend_songs

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vibe2Music",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --bg2:       #12121a;
    --bg3:       #1a1a26;
    --accent:    #7c6af7;
    --accent2:   #f76a8a;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --kaggle:    #7c6af7;
    --lastfm:    #f76a8a;
    --none:      #4a4a5a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Ocultar elementos de Streamlit que no necesitamos */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* Título principal */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -2px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
}

/* Input de vibe */
[data-testid="stTextInput"] input {
    background: var(--bg3) !important;
    border: 1.5px solid var(--bg3) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.15) !important;
}

/* Botón principal */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Tarjetas de canciones */
.song-card {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s, transform 0.15s;
}
.song-card:hover {
    border-color: var(--accent);
    transform: translateX(4px);
}
.song-rank {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--muted);
    min-width: 2rem;
    text-align: center;
}
.song-info { flex: 1; min-width: 0; }
.song-name {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-artist {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 2px;
}
.song-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    text-align: right;
    min-width: 3rem;
}

/* Badges de fuente */
.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}
.badge-kaggle  { background: rgba(124,106,247,0.2); color: var(--kaggle); }
.badge-lastfm  { background: rgba(247,106,138,0.2); color: var(--lastfm); }
.badge-none    { background: rgba(74,74,90,0.3);    color: var(--muted);  }

/* Sección de visualización */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem;
}

/* Métricas de cobertura */
.coverage-box {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.coverage-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}
.coverage-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 2px;
}

/* Plotly charts */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* Sliders */
[data-testid="stSlider"] > div > div { color: var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--bg3) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return load_dataset()

df = load_data()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>VIBE2MUSIC</h1>
    <p>describe cómo te sientes · recibe tu soundtrack</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ── Input principal ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])

with col_input:
    vibe = st.text_input(
        label="vibe",
        placeholder="ej: música tranquila para estudiar de noche, gym motivation, fiesta reggaeton...",
        label_visibility="collapsed",
    )

with col_btn:
    buscar = st.button("▶ GENERAR", use_container_width=True)


# ── Opciones avanzadas ─────────────────────────────────────────────────────────
with st.expander("⚙ opciones avanzadas"):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        top_n = st.slider("Número de canciones", 5, 20, 10)
    with col_b:
        max_artist = st.slider("Máx. por artista", 1, 5, 2)
    with col_c:
        w_sem = st.slider("Peso semántico", 0.0, 1.0, 0.65, 0.05)
    with col_d:
        w_ac = st.slider("Peso acústico", 0.0, 1.0, 0.25, 0.05)


# ── Resultados ─────────────────────────────────────────────────────────────────
if buscar and vibe:
    with st.spinner("analizando tu vibe..."):
        results = recommend_songs(
            vibe, df,
            top_n=top_n,
            max_per_artist=max_artist,
            weight_semantic=w_sem,
            weight_acoustic=w_ac,
            weight_tags=max(0.0, 1.0 - w_sem - w_ac),
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Layout: lista izquierda, charts derecha ────────────────────────────────
    col_list, col_viz = st.columns([1, 1.4], gap="large")

    # ── Lista de canciones ─────────────────────────────────────────────────────
    with col_list:
        st.markdown('<p class="section-title">🎵 tus canciones</p>', unsafe_allow_html=True)

        for i, row in results.iterrows():
            src = row.get("audio_source", "none")
            badge_class = f"badge-{src.replace('_tags','')}" if src != "none" else "badge-none"
            badge_text  = {"kaggle": "kaggle", "lastfm_tags": "last.fm", "none": "sem"}.get(src, src)
            score_pct   = f"{row['similarity']*100:.0f}%"

            st.markdown(f"""
            <div class="song-card">
                <div class="song-rank">{list(results.index).index(i)+1:02d}</div>
                <div class="song-info">
                    <div class="song-name">{row['name']}</div>
                    <div class="song-artist">{row['artist']} &nbsp;
                        <span class="badge {badge_class}">{badge_text}</span>
                    </div>
                </div>
                <div class="song-score">{score_pct}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Visualizaciones ────────────────────────────────────────────────────────
    with col_viz:

        # ── Scatter: espacio emocional (valence vs energy) ─────────────────────
        has_features = results[results["audio_source"] == "kaggle"].copy()

        if len(has_features) >= 2 and "energy" in has_features.columns:
            st.markdown('<p class="section-title">📍 espacio emocional</p>', unsafe_allow_html=True)

            # Cuadrantes del modelo circumplex de Russell (1980)
            fig_scatter = go.Figure()

            # Fondo de cuadrantes
            quadrant_style = dict(
                layer="below", line_width=0, opacity=0.06
            )
            fig_scatter.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1,
                fillcolor="#7c6af7", **quadrant_style)
            fig_scatter.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1,
                fillcolor="#f76a8a", **quadrant_style)
            fig_scatter.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5,
                fillcolor="#4a9eff", **quadrant_style)
            fig_scatter.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                fillcolor="#4affb0", **quadrant_style)

            # Etiquetas de cuadrantes
            for x, y, label, color in [
                (0.25, 0.92, "tenso / angustiado", "#7c6af7"),
                (0.75, 0.92, "feliz / eufórico",   "#f76a8a"),
                (0.25, 0.08, "triste / deprimido",  "#4a9eff"),
                (0.75, 0.08, "tranquilo / relajado","#4affb0"),
            ]:
                fig_scatter.add_annotation(
                    x=x, y=y, text=label,
                    showarrow=False,
                    font=dict(size=9, color=color, family="Space Mono"),
                    opacity=0.7,
                )

            # Líneas de ejes
            for val in [0.5]:
                fig_scatter.add_shape(type="line", x0=val, y0=0, x1=val, y1=1,
                    line=dict(color="#ffffff", width=0.5, dash="dot"))
                fig_scatter.add_shape(type="line", x0=0, y0=val, x1=1, y1=val,
                    line=dict(color="#ffffff", width=0.5, dash="dot"))

            # Puntos de canciones
            fig_scatter.add_trace(go.Scatter(
                x=has_features["valence"].fillna(0.5),
                y=has_features["energy"].fillna(0.5),
                mode="markers+text",
                text=has_features["name"].apply(lambda x: x[:18] + "…" if len(x) > 18 else x),
                textposition="top center",
                textfont=dict(size=9, color="#e8e8f0", family="DM Sans"),
                marker=dict(
                    size=has_features["similarity"] * 18 + 6,
                    color=has_features["similarity"],
                    colorscale=[[0, "#4a4a5a"], [0.5, "#7c6af7"], [1, "#f76a8a"]],
                    showscale=False,
                    line=dict(color="#0a0a0f", width=1.5),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Valence: %{x:.2f}<br>"
                    "Energy: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

            fig_scatter.update_layout(
                xaxis=dict(title="Valence →  (negativo → positivo)",
                           range=[0, 1], gridcolor="#1a1a26",
                           title_font=dict(size=10, color="#6b6b80"),
                           tickfont=dict(size=9, color="#6b6b80")),
                yaxis=dict(title="Energy →  (tranquilo → intenso)",
                           range=[0, 1], gridcolor="#1a1a26",
                           title_font=dict(size=10, color="#6b6b80"),
                           tickfont=dict(size=9, color="#6b6b80")),
                plot_bgcolor="#0a0a0f",
                paper_bgcolor="#12121a",
                margin=dict(l=10, r=10, t=10, b=10),
                height=340,
                font=dict(family="DM Sans"),
                showlegend=False,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Radar: perfil acústico medio ───────────────────────────────────────
        radar_cols_wanted = ["danceability", "energy", "valence",
                             "acousticness", "instrumentalness", "speechiness"]
        # Solo usar columnas que realmente existen en el dataframe de resultados
        radar_cols = [c for c in radar_cols_wanted if c in has_features.columns]
        radar_data = has_features[radar_cols].dropna() if (len(has_features) > 0 and radar_cols) else pd.DataFrame()

        if len(radar_data) >= 2:
            st.markdown('<p class="section-title">📊 perfil acústico medio</p>', unsafe_allow_html=True)

            means = radar_data.mean()
            labels_es = {
                "danceability": "bailabilidad",
                "energy": "energía",
                "valence": "positividad",
                "acousticness": "acústico",
                "instrumentalness": "instrumental",
                "speechiness": "vocal/rap",
            }
            categories = [labels_es[c] for c in radar_cols]
            values = [means[c] for c in radar_cols]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(124, 106, 247, 0.15)",
                line=dict(color="#7c6af7", width=2),
                name="perfil medio",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0a0a0f",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        gridcolor="#1a1a26", tickfont=dict(size=8, color="#6b6b80"),
                        tickvals=[0.25, 0.5, 0.75],
                    ),
                    angularaxis=dict(
                        gridcolor="#1a1a26",
                        tickfont=dict(size=10, color="#e8e8f0", family="DM Sans"),
                    ),
                ),
                paper_bgcolor="#12121a",
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Score breakdown ────────────────────────────────────────────────────
        st.markdown('<p class="section-title">🔍 desglose de scores</p>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        song_labels = results.apply(
            lambda r: f"{r['name'][:20]}… / {r['artist'][:12]}" if len(r['name']) > 20
            else f"{r['name']} / {r['artist'][:12]}", axis=1
        )

        for col, color, label in [
            ("similarity_semantic", "#7c6af7", "semántico"),
            ("similarity_acoustic", "#f76a8a", "acústico"),
        ]:
            if col in results.columns:
                fig_bar.add_trace(go.Bar(
                    name=label,
                    x=results[col].values,
                    y=song_labels.values,
                    orientation="h",
                    marker_color=color,
                    opacity=0.85,
                    text=[f"{v:.0%}" for v in results[col].values],
                    textposition="inside",
                    textfont=dict(size=9, color="white"),
                ))

        fig_bar.update_layout(
            barmode="group",
            xaxis=dict(range=[0, 1], gridcolor="#1a1a26",
                       tickfont=dict(size=9, color="#6b6b80"), showgrid=True),
            yaxis=dict(tickfont=dict(size=9, color="#e8e8f0"), autorange="reversed"),
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#12121a",
            legend=dict(orientation="h", y=1.05,
                        font=dict(size=10, color="#6b6b80")),
            margin=dict(l=10, r=10, t=30, b=10),
            height=max(250, top_n * 28),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Cobertura del dataset ──────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">📦 fuentes de datos en estas recomendaciones</p>',
                unsafe_allow_html=True)

    src_counts = results["audio_source"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    cols_map = {
        "kaggle":      (c1, "#7c6af7", "Kaggle", "features acústicas"),
        "lastfm_tags": (c2, "#f76a8a", "Last.fm", "tags de género"),
        "none":        (c3, "#4a4a5a", "Solo semántico", "sin features"),
    }
    total_shown = len(results)
    for src, (col, color, name, desc) in cols_map.items():
        n = src_counts.get(src, 0)
        with col:
            st.markdown(f"""
            <div class="coverage-box">
                <div class="coverage-num" style="color:{color}">{n}</div>
                <div class="coverage-label" style="color:{color};font-weight:600">{name}</div>
                <div class="coverage-label">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="coverage-box">
            <div class="coverage-num" style="color:#e8e8f0">{total_shown}</div>
            <div class="coverage-label" style="font-weight:600">Total</div>
            <div class="coverage-label">canciones recomendadas</div>
        </div>
        """, unsafe_allow_html=True)

elif buscar and not vibe:
    st.warning("Escribe una vibe primero.")

else:
    # Estado vacío
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0; color: #3a3a4a;'>
        <div style='font-size:3rem'>🎧</div>
        <div style='font-family: Space Mono, monospace; font-size:0.8rem;
                    letter-spacing:2px; margin-top:1rem;'>
            ESCRIBE TU VIBE Y PULSA GENERAR
        </div>
    </div>
    """, unsafe_allow_html=True)