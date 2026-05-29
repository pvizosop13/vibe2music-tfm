from sentence_transformers import SentenceTransformer

# Modelo multilingüe: misma arquitectura MiniLM pero entrenado en 50 idiomas.
# Permite comparar vibes en español con canciones en inglés y viceversa
# sin pérdida de precisión semántica por cruce de idioma.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text):
    return model.encode(text)