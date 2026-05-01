import streamlit as st
from src.model.recommender import load_dataset, recommend_songs

# configuración de la página
st.set_page_config(page_title="Vibe2Music", layout="centered")

st.title("🎧 Vibe2Music")
st.subheader("Convierte tu vibe en música")
st.markdown("Escribe cómo te sientes o qué quieres escuchar, y te recomendaré canciones de TU biblioteca")
# cargar dataset
@st.cache_data
def load_data():
    return load_dataset()

df = load_data()

# input del usuario
vibe = st.text_input("Describe tu vibe (ej: chill study, gym motivation...)")

# botón
if st.button("Generar recomendaciones"):
    if vibe:
        results = recommend_songs(vibe, df, top_n=10)
        
        st.success("Aquí tienes tus canciones:")
        
        for i, row in results.iterrows():
            st.write(f"🎵 {row['name']} - {row['artist']}")
    else:
        st.warning("Por favor, introduce una vibe")